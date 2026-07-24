import os
import pickle
import numpy as np
from copy import deepcopy

from datasets.base import CBCT_dataset


class CBCT_dataset_CT(CBCT_dataset):
    def __getitem__(self, index):
        name = self.name_list[index]

        ct, _, _ = self.load_ct(name, no_scale=True)
        ct = ct[None, ...]

        # -- load sampling points
        if not self.is_train:
            points = deepcopy(self.points)
            points_gt, spacing, origin = self.load_ct(name)
        else:
            b_idx = np.random.randint(len(self.blocks))
            block_values = self.load_block(name, b_idx)
            block_coords = self.blocks[b_idx] # [N, 3]
            points, points_gt = self.sample_points(block_coords, block_values)
            points_gt = points_gt.astype(np.float32) / 255.
            points_gt = points_gt[None, :]

        points = (points - 0.5) * 2

        # -- collect data
        ret_dict = {
            # M: the number of views
            # N: the number of sampled points
            'dst_name': self.dst_name,
            'name': name,
            'ct': ct,               # [1, W, H, D]
            'points_ct': points,    # [N, 3], aligned with CT
            'points_gt': points_gt, # [1, N] (or [W', H', D'] only when is_train is False)
        }
        if not self.is_train:
            ret_dict['spacing'] = spacing # [3,]
            ret_dict['origin'] = origin   # [3,]
        return ret_dict
