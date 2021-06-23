import torch
import torch.nn as nn
import model.common as common
import numpy as np
from scipy import ndimage
import torch.nn.functional as F
import model.utils as utils
import random


def make_model(args, parent=False):
    return CNSR(args)


class CorrectionRefine(nn.Module):
    def __init__(self, args, act_mode='R'):
        super(CorrectionRefine, self).__init__()

        self.m_head = common.conv(args.n_colors, args.n_feats, bias=True, mode='C' + act_mode)

        self.m_body = common.sequential(*[common.ResBlock(args.n_feats, args.n_feats, bias=True, mode='C' + act_mode + 'C')
                                          for _ in range(args.n_resblocks)])

        self.m_tail = common.conv(args.n_feats, args.n_colors, bias=True, mode='C')

    def forward(self, x):

        x_h = self.m_head(x)
        x_b = self.m_body(x_h)
        x_t = self.m_tail(x_b)

        return x + x_t


class CorrectionFilter(nn.Module):
    def __init__(self, args):
        super(CorrectionFilter, self).__init__()
        self.scale_factor = args.scale[0]
        self.gpu = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.gpu)
        self.r = utils.get_bicubic(self.scale_factor).float().to(self.device)
        self.r = self.r / self.r.sum()
        self.shape = None

        # self.eps = 1e-3   # scale factor 4
        if self.scale_factor == 2:
            self.eps = 5e-3  # scale factor 2
        else:
            self.eps = 1e-4  # scale factor 2
        self.inv_type = 'Tikhonov'

    def find_H(self, s, r):
        R = utils.fft_torch(r, self.shape)
        S = utils.fft_torch(s, self.shape)

        R, S = utils.shift_by(R, 0.5 * (not self.scale_factor % 2)), utils.shift_by(S,
                                                                                    0.5 * (not self.scale_factor % 2))

        # Find Q = S*R
        Q = S.conj() * R
        q = torch.fft.ifftn(Q, dim=(-2, -1))

        q_d = q[:, :, 0::self.scale_factor, 0::self.scale_factor]
        Q_d = torch.fft.fftn(q_d, dim=(-2, -1))

        # Find R*R
        RR = R.conj() * R
        rr = torch.fft.ifftn(RR, dim=(-2, -1))
        rr_d = rr[:, :, 0::self.scale_factor, 0::self.scale_factor]
        RR_d = torch.fft.fftn(rr_d, dim=(-2, -1))

        # Invert S*R
        Q_d_inv = utils.dagger(Q_d, self.eps, mode=self.inv_type)

        H = RR_d * Q_d_inv

        return H

    def forward(self, x, s):
        s = torch.roll(s, (-s.shape[2]//2, -s.shape[3]//2), dims=(2,3))
        self.shape = (x.shape[2]*self.scale_factor, x.shape[3]*self.scale_factor)
        out = torch.zeros(x.shape).to(self.device)
        for i in range(out.shape[0]):
            H = self.find_H(s[i:i+1,...]/s[i:i+1,...].sum(), self.r)
            out[i:i+1,...] = utils.fft_Filter_(x[i:i+1,...], H)
        return out


class KernelEstimate(nn.Module):
    def __init__(self, args, act_mode='R'):
        super(KernelEstimate, self).__init__()

        self.m_head = common.conv(args.n_colors, args.n_feats, bias=True, mode='C'+ act_mode)

        self.m_body = common.sequential(
            *[common.ResBlock(args.n_feats, args.n_feats, bias=True, mode='C' + act_mode + 'C') for _ in range(args.n_resblocks)])

        self.m_tail = common.conv(args.n_feats, args.n_kernels, bias=True, mode='C')

    def forward(self, x):

        x = self.m_head(x)
        x = self.m_body(x)
        x = self.m_tail(x)

        return x


class CNSR(nn.Module):
    def __init__(self, args):
        super(CNSR, self).__init__()
        self.p = CorrectionRefine(args)
        self.d = CorrectionFilter(args)
        self.k = KernelEstimate(args)
        # self.gpu = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = torch.device(self.gpu)
        # state_dict1 = torch.load(
        #     '/mnt/ori_home/caoxiang112/KERNEL/experiment/kernelestimate_x2_32_nb14_epoch1000_loss/model/model_best.pt',
        #     map_location=self.gpu)
        # self.k.load_state_dict(state_dict1)
        # self.k.eval()
        # for k, v in self.k.named_parameters():
        #     v.requires_grad = False
        # self.k = self.k.to(self.device)

    def forward(self, x):

        H, W = x.shape[2:]
        k = self.k(x[:, :, H//2-16:H//2+16, W//2-16:W//2+16])
        x = self.d(x, k)
        x = self.p(x)

        return x
