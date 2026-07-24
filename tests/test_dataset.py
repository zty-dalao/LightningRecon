import builtins
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk
import torch

from src.dataset import PairedCBCTDataset
from src.train import subsample_projections


class PairedCBCTDatasetFallbackTest(unittest.TestCase):
    def test_subsample_projections_uses_actual_view_count(self):
        projs = torch.randn(1, 490, 1, 8, 8)
        sampled = subsample_projections(projs, 24, torch.device('cpu'))

        self.assertEqual(sampled.shape, (1, 24, 1, 8, 8))

    def test_loads_volume_with_simpleitk_when_nibabel_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proj_dir = os.path.join(tmpdir, 'projection_npy', 'case0')
            cbct_dir = os.path.join(tmpdir, 'cbct')
            ct_dir = os.path.join(tmpdir, 'ct')
            os.makedirs(proj_dir, exist_ok=True)
            os.makedirs(cbct_dir, exist_ok=True)
            os.makedirs(ct_dir, exist_ok=True)

            np.save(os.path.join(proj_dir, 'Proj_00000.npy'), np.zeros((8, 16), dtype=np.float32))

            arr = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
            sitk.WriteImage(sitk.GetImageFromArray(arr), os.path.join(cbct_dir, 'case0.nii.gz'))
            sitk.WriteImage(sitk.GetImageFromArray(arr), os.path.join(ct_dir, 'case0.nii.gz'))

            original_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == 'nibabel':
                    raise ModuleNotFoundError("No module named 'nibabel'")
                return original_import(name, *args, **kwargs)

            with patch('builtins.__import__', side_effect=fake_import):
                dataset = PairedCBCTDataset(
                    data_root=tmpdir,
                    split='train',
                    n_views=1,
                    proj_size=(16, 16),
                    vol_size=(4, 4, 4),
                    val_ratio=0.0,
                )
                sample = dataset[0]

            self.assertEqual(sample['projs'].shape, (1, 1, 16, 16))
            self.assertEqual(sample['cbct'].shape, (1, 4, 4, 4))
            self.assertEqual(sample['ct'].shape, (1, 4, 4, 4))


if __name__ == '__main__':
    unittest.main()
