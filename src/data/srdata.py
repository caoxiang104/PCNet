import os
import glob
import random
import pickle

from data import common

import numpy as np
import imageio
import torch
import torch.utils.data as data
from scipy import ndimage


class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale
        self.idx_scale = 0

        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_lr, list_hr = self._scan()
        if args.ext.find('img') >= 0 or benchmark:
            self.images_lr, self.images_hr = list_lr, list_hr
        elif args.ext.find('sep') >= 0:
            os.makedirs(
                self.dir_lr.replace(self.apath, path_bin),
                exist_ok=True
            )
            os.makedirs(
                self.dir_hr.replace(self.apath, path_bin),
                exist_ok=True
            )

            self.images_lr, self.images_hr = [], []
            for h in list_lr:
                b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[0], '.pt')
                self.images_lr.append(b)
                self._check_and_load(args.ext, h, b, verbose=True)
            for h in list_hr:
                b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[1], '.pt')
                self.images_hr.append(b)
                self._check_and_load(args.ext, h, b, verbose=True)
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_lr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    # Below functions as used to prepare images
    def _scan(self):
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '*' + self.ext[0]))
        )
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[1]))
        )

        return names_lr, names_hr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_lr = os.path.join(self.apath, 'LR_gus_2x')
        self.dir_hr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('.png', '.png')

    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1], filename

    def __len__(self):
        if self.train:
            return len(self.images_lr) * self.repeat
        else:
            return len(self.images_lr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_lr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_lr = self.images_lr[idx]
        f_hr = self.images_hr[idx]

        filename, _ = os.path.splitext(os.path.basename(f_lr))
        if self.args.ext == 'img' or self.benchmark:
            lr = imageio.imread(f_lr)
            hr = imageio.imread(f_hr)
        elif self.args.ext.find('sep') >= 0:
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)

        return lr, hr, filename

    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        if self.train:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(len(self.scale) > 1),
                input_large=self.input_large
            )
            if not self.args.no_augment: lr, hr = common.augment(lr, hr)

        return lr/255.0, hr/255.0

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)

