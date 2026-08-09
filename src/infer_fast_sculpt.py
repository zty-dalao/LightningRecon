"""载入两个checkpoint，以固定6/8/10视角输出最终256³ sCT。"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from src.fast_sculpt import BaseSCTNet, ProjectionGuidedSculptor
from src.thorax_fast_dataset import ThoraxFastDataset
from src.view_protocol import resolve_view_curriculum


@torch.inference_mode()
def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1 = torch.load(args.stage1_checkpoint, map_location=device,
                        weights_only=False)
    stage2 = torch.load(args.stage2_checkpoint, map_location=device,
                        weights_only=False)
    base_model = BaseSCTNet(**stage1.get(
        "model_config", {"base_channels": 4}
    )).to(device).eval()
    base_model.load_state_dict(stage1["model"])
    sculptor = ProjectionGuidedSculptor(**stage2["model_config"]).to(device).eval()
    sculptor.load_state_dict(stage2["model"])

    base_views = resolve_view_curriculum(args.final_view)[0]
    dataset = ThoraxFastDataset(
        data_root=args.data_root, split=args.split,
        volume_keys=("cbct",), projection_views=args.final_view,
        final_view=args.final_view, projection_base_views=base_views,
        projection_sampling="nested_uniform",
        projection_size=tuple(args.projection_size),
        volume_size=tuple(args.volume_size), require_projections=True,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample in dataset:
        cbct = sample["cbct"].unsqueeze(0).to(device)
        projs = sample["projs"].unsqueeze(0).to(device)
        angles = sample["angles"].unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            base = base_model(cbct)["base_sct"]
            final = sculptor(base, projs, angles)["final_sct"]
        # 模型(D,H,W)恢复成NIfTI的(X,Y,Z)，保存归一化[0,1]结果。
        array = final[0, 0].float().cpu().numpy().transpose(2, 1, 0)
        affine = sample["cbct_affine"].numpy()
        nib.save(nib.Nifti1Image(array.astype(np.float32), affine),
                 output_dir / f"{sample['case_id']}_sct.nii.gz")
        print(f"saved {sample['case_id']}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--stage2_checkpoint", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split", choices=("train","val","eval","test"),
                        default="test")
    parser.add_argument("--final_view", type=int, choices=(6,8,10), default=6)
    parser.add_argument("--output_dir", default="outputs/fast_sculpt")
    parser.add_argument("--projection_size", type=int, nargs=2, default=(128,128))
    parser.add_argument("--volume_size", type=int, nargs=3, default=(256,256,256))
    return parser


if __name__ == "__main__":
    infer(build_parser().parse_args())
