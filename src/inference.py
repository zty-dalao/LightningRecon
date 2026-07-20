"""
推理脚本: 从稀疏投影直接重建 CT 体素。

流程:
  稀疏投影 + 角度编码 → CNN → Transformer → 双码本 → 渐进解码 → 256³ NIfTI

用法:
  python src/inference.py --checkpoint logs/thorax_6view/best_model.pth \
      --data_root data/thorax_fast --case_id CASE --n_views 6 --output recon.nii.gz
"""

import os, sys, argparse
import torch, numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SparseViewReconstruction
from src.dataset import PairedCBCTDataset


def add_angle_encoding(projs, device):
    """投影 + sin/cos 角度编码 → (V, 3, H, W)"""
    V, H, W = projs.shape
    theta = torch.linspace(0, 2 * torch.pi, V + 1, device=device)[:V]
    sin_map = torch.sin(theta).view(-1, 1, 1, 1).expand(-1, 1, H, W)
    cos_map = torch.cos(theta).view(-1, 1, 1, 1).expand(-1, 1, H, W)
    return torch.cat([projs.unsqueeze(1), sin_map, cos_map], dim=1)  # (V, 3, H, W)


def load_model(checkpoint_path, device, n_decoder_ups=1):
    model = SparseViewReconstruction(n_decoder_ups=n_decoder_ups).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'Loaded epoch {ckpt.get("epoch","?")}, PSNR={ckpt.get("best_psnr","N/A")}')
    return model


@torch.no_grad()
def reconstruct_single(model, dataset, case_id, device, n_views=6,
                       ct_range=(-1000, 1000)):
    try:
        idx = dataset.cases.index(case_id)
    except ValueError:
        raise ValueError(f'Case "{case_id}" not found.')

    batch = dataset[idx]
    projs = batch['projs'][:n_views, 0, :, :]                       # (V, H, W)
    projs_enc = add_angle_encoding(projs.to(device), device)         # (V, 3, H, W)
    projs_enc = projs_enc.unsqueeze(0)                               # (1, V, 3, H, W)

    pred, _, _ = model(projs_enc)
    ct_min, ct_max = ct_range
    volume = pred[0, 0].cpu().numpy()
    volume = volume * (ct_max - ct_min) + ct_min
    return volume.astype(np.float32)


def save_nifti(volume, path, spacing=(0.98, 0.98, 2.0)):
    vol = np.transpose(volume, (2, 1, 0))
    aff = np.eye(4)
    aff[0,0], aff[1,1], aff[2,2] = spacing
    nib.save(nib.Nifti1Image(vol, aff), path)
    print(f'Saved {path}')


def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device)
    n_views = args.n_views or 6
    dataset = PairedCBCTDataset(data_root=args.data_root,
                                split=args.split if args.case_id is None else 'test',
                                n_views=n_views, proj_size=(256, 256),
                                vol_size=(128, 128, 128))
    print(f'Dataset: {len(dataset)} cases, {n_views} views')

    if args.case_id:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        vol = reconstruct_single(model, dataset, args.case_id, device, n_views, args.ct_range)
        save_nifti(vol, args.output)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        for cid in dataset.cases:
            print(f'Processing: {cid}')
            vol = reconstruct_single(model, dataset, cid, device, n_views, args.ct_range)
            save_nifti(vol, os.path.join(args.output_dir, f'{cid}_recon.nii.gz'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--case_id', type=str, default=None)
    p.add_argument('--split', type=str, default='test')
    p.add_argument('--n_views', type=int, default=6)
    p.add_argument('--output', type=str, default='./recon.nii.gz')
    p.add_argument('--output_dir', type=str, default='./outputs/')
    p.add_argument('--ct_range', type=int, nargs=2, default=(-1000, 1000))
    args = p.parse_args()
    inference(args)
