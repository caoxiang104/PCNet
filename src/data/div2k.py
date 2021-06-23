import os
from data import srdata


class DIV2K(srdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(DIV2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_lr, names_hr = super(DIV2K, self)._scan()
        names_lr = names_lr[self.begin - 1:self.end]
        names_hr = names_hr[self.begin - 1:self.end]

        return names_lr, names_hr

    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data)
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_gus_blurX2')
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_LR_bic_X2')


