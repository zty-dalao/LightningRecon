import os
import pickle
import numpy as np
from copy import deepcopy

from datasets.base import CBCT_dataset



class CBCT_dataset_LR(CBCT_dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        lr_res = self.cfg.lr_res
        points_lr = np.mgrid[:lr_res, :lr_res, :lr_res] / lr_res # [3, ...], ~[0, 1]
        points_lr = points_lr.reshape(3, -1).transpose(1, 0) # [N, 3]
        self.points_lr = points_lr

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)
        points_lr_proj = self.project_points(self.points_lr, data_dict['angles'])
        points_ct = deepcopy(data_dict['points'])
        points_ct = (points_ct - 0.5) * 2

        data_dict.update({
            'points_lr_proj': points_lr_proj,
            'points_ct': points_ct # [N, 3]
        })
        return data_dict

class CBCT_dataset_LR_CT(CBCT_dataset_LR):
    def __getitem__(self, index):
        data_dict = super().__getitem__(index)
        
        name = data_dict['name']
        ct, _, _ = self.load_ct(name, no_scale=True)
        ct = ct[None, ...]

        data_dict.update({
            'ct': ct,
        })
        return data_dict

