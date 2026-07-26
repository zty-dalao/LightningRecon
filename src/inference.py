"""
推理脚本: 从稀疏投影直接重建 CT 体素。

流程:
  稀疏投影 + 角度编码 → CNN → Transformer → 双码本 → 渐进解码 → NIfTI

用法:
  python src/inference.py --checkpoint \
      'logs/thorax_fast_finalview=6_256_v3/best_model_finalview=6_v3.pth' \
      --data_root ~/autodl-tmp/thorax --case_id 2026-06-04_065713 --final_view 6 --output recon.nii.gz
"""

import os, sys, argparse
import torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SparseViewReconstruction
from src.dataset import ThoraxCTDataset
from src.view_protocol import (
    resolve_view_curriculum,
    uniform_view_indices as protocol_view_indices,
)


def uniform_view_indices(total_views, n_views, device):
    return torch.tensor(
        protocol_view_indices(total_views,n_views),
        device=device,
        dtype=torch.long,
    )


def add_angle_encoding(projs, angles, device):
    """投影 + sin/cos 角度编码 → (V, 3, H, W)"""
    V, H, W = projs.shape
    if angles.shape != (V,):
        raise ValueError(f'Expected {V} angles, got {tuple(angles.shape)}')
    theta = angles.to(device=device, dtype=projs.dtype)
    sin_map = torch.sin(theta).view(-1, 1, 1, 1).expand(-1, 1, H, W)
    cos_map = torch.cos(theta).view(-1, 1, 1, 1).expand(-1, 1, H, W)
    return torch.cat([projs.unsqueeze(1), sin_map, cos_map], dim=1)  # (V, 3, H, W)


def validate_checkpoint_metadata(checkpoint):
    """Reject legacy checkpoints that cannot reproduce preprocessing safely."""
    if int(checkpoint.get('checkpoint_format', 0)) != 3:
        raise ValueError(
            'Unsupported legacy checkpoint. Expected checkpoint_format=3 '
            'with source-grid, model, and HU metadata.'
        )
    required = (
        'model_state','model_config','ct_normalization','source_views',
        'eval_views','final_view','run_version',
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(
            f'Checkpoint is missing required metadata: {missing}'
        )
    return checkpoint


def load_model(checkpoint_path, device, n_decoder_ups=None,
               transformer_layers=None,
               expected_views=None, expected_source_views=None,
               checkpoint_data=None):
    ckpt = (
        checkpoint_data
        if checkpoint_data is not None
        else torch.load(checkpoint_path, map_location='cpu')
    )
    validate_checkpoint_metadata(ckpt)
    model_config = ckpt['model_config']
    saved_ups = int(model_config['n_decoder_ups'])
    # Early format-3 checkpoints predate this field and used two layers.
    saved_transformer_layers = int(model_config.get('transformer_layers', 2))
    if n_decoder_ups is not None and int(n_decoder_ups) != saved_ups:
        raise ValueError(
            f'n_decoder_ups={n_decoder_ups} does not match checkpoint '
            f'n_decoder_ups={saved_ups}'
        )
    if (
        transformer_layers is not None
        and int(transformer_layers) != saved_transformer_layers
    ):
        raise ValueError(
            f'transformer_layers={transformer_layers} does not match '
            f'checkpoint transformer_layers={saved_transformer_layers}'
        )
    model = SparseViewReconstruction(
        n_decoder_ups=saved_ups,
        transformer_layers=saved_transformer_layers,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    metric=ckpt.get('best_mask_psnr','N/A')
    eval_views=ckpt['eval_views']
    source_views=ckpt['source_views']
    run_version=ckpt['run_version']
    if expected_views is not None:
        if int(eval_views) != int(expected_views):
            raise ValueError(
                f'Checkpoint was selected at {eval_views} views, '
                f'but inference requested {expected_views}'
            )
    if expected_source_views is not None:
        if int(source_views) != int(expected_source_views):
            raise ValueError(
                f'Checkpoint uses a {source_views}-view source grid, '
                f'but inference requested {expected_source_views}'
            )
    print(
        f'Loaded epoch {ckpt.get("epoch","?")}, '
        f'Mask PSNR={metric}, eval_views={eval_views}, '
        f'source_views={source_views}, run_version={run_version}, '
        f'transformer_layers={saved_transformer_layers}'
    )
    return model,ckpt


@torch.no_grad()
def reconstruct_single(model, dataset, case_id, device, n_views=6,
                       ct_range=(-1000, 1000)):
    try:
        idx = dataset.cases.index(case_id)
    except ValueError:
        raise ValueError(f'Case "{case_id}" not found.')

    batch = dataset[idx]
    source_projs = batch['projs'][:, 0, :, :]
    source_angles = batch['angles']
    expected_indices = uniform_view_indices(
        dataset.max_views,n_views,batch['view_indices'].device
    )
    if not torch.equal(batch['view_indices'],expected_indices):
        raise ValueError(
            f'Case {case_id!r} does not contain the expected direct '
            f'{dataset.max_views}->{n_views} source-grid protocol'
        )
    projs = source_projs
    angles = source_angles
    projs_enc = add_angle_encoding(projs.to(device),angles,device)   # (V, 3, H, W)
    projs_enc = projs_enc.unsqueeze(0)                               # (1, V, 3, H, W)

    pred, _, _ = model(projs_enc)
    ct_min, ct_max = ct_range
    normalized = pred[0, 0].float()
    outside_fraction = float(
        ((normalized < 0.0) | (normalized > 1.0)).float().mean()
    )
    if outside_fraction > 0:
        print(
            f'Warning: {outside_fraction:.2%} predicted voxels were outside '
            '[0,1] and were clipped before HU conversion.'
        )
    volume = normalized.clamp(0.0, 1.0).cpu().numpy()
    volume = volume * (ct_max - ct_min) + ct_min
    return volume.astype(np.float32)


def resampled_affine(reference_affine, reference_shape, output_shape):
    """Preserve reference orientation/FOV for a differently sized output grid."""
    reference_shape = np.asarray(reference_shape[:3], dtype=np.float64)
    output_shape = np.asarray(output_shape[:3], dtype=np.float64)
    scale = reference_shape / output_shape
    voxel_transform = np.eye(4, dtype=np.float64)
    voxel_transform[0,0],voxel_transform[1,1],voxel_transform[2,2] = scale
    # align_corners=False-style center alignment
    voxel_transform[:3,3] = 0.5 * scale - 0.5
    return np.asarray(reference_affine) @ voxel_transform


def save_nifti(volume, path, reference_path):
    import nibabel as nib

    vol = np.transpose(volume, (2, 1, 0))
    reference = nib.load(reference_path)
    affine = resampled_affine(
        reference.affine,reference.shape,vol.shape
    )
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0,0.0)
    header['cal_min'] = float(np.min(vol))
    header['cal_max'] = float(np.max(vol))
    nib.save(
        nib.Nifti1Image(vol.astype(np.float32),affine,header=header),path
    )
    print(f'Saved {path}')


def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resolve_view_curriculum(
        args.final_view,args.view_schedule,max_views=64
    )
    n_views=args.final_view
    checkpoint_preview=validate_checkpoint_metadata(
        torch.load(args.checkpoint,map_location='cpu')
    )
    source_views=int(checkpoint_preview['source_views'])
    if args.source_views is not None and args.source_views != source_views:
        raise ValueError(
            f'--source_views={args.source_views} does not match checkpoint '
            f'source_views={source_views}'
        )
    model_config=checkpoint_preview['model_config']
    saved_proj_size=tuple(model_config['proj_size'])
    saved_vol_size=tuple(model_config['vol_size'])
    if args.proj_size is not None and tuple(args.proj_size)!=saved_proj_size:
        raise ValueError(
            f'--proj_size={tuple(args.proj_size)} does not match checkpoint '
            f'proj_size={saved_proj_size}'
        )
    ct_metadata=checkpoint_preview['ct_normalization']
    saved_ct_range=tuple(ct_metadata['hu_range'])
    ct_range=tuple(args.ct_range) if args.ct_range is not None else saved_ct_range
    if ct_range[0] >= ct_range[1]:
        raise ValueError(f'ct_range must be increasing, got {ct_range}')
    model,_ = load_model(
        args.checkpoint,device,n_decoder_ups=args.n_decoder_ups,
        transformer_layers=args.transformer_layers,
        expected_views=n_views,
        expected_source_views=source_views,
        checkpoint_data=checkpoint_preview,
    )
    dataset = ThoraxCTDataset(data_root=args.data_root,
                             split=args.split if args.case_id is None else 'test',
                             n_views=-1,
                             view_counts=(n_views,),
                             proj_size=saved_proj_size,
                             vol_size=saved_vol_size,
                             expected_source_views=source_views)
    print(
        f'Dataset: {len(dataset)} cases, '
        f'{source_views}-view source grid -> {n_views}-view inference'
    )

    if args.case_id:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        vol = reconstruct_single(
            model,dataset,args.case_id,device,n_views,ct_range
        )
        save_nifti(vol,args.output,dataset.ct_path(args.case_id))
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        for cid in dataset.cases:
            print(f'Processing: {cid}')
            vol = reconstruct_single(
                model,dataset,cid,device,n_views,ct_range
            )
            save_nifti(
                vol,os.path.join(args.output_dir,f'{cid}_recon.nii.gz'),
                dataset.ct_path(cid),
            )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--case_id', type=str, default=None)
    p.add_argument('--split', type=str, default='test')
    p.add_argument('--final_view', type=int, choices=(6,8,10), default=6)
    p.add_argument('--view_schedule', type=str, default=None,
                   help='Optional protocol validation; final_view controls inference')
    p.add_argument('--source_views', type=int, default=None)
    p.add_argument('--proj_size', type=int, nargs=2, default=None)
    p.add_argument('--n_decoder_ups', type=int, default=None)
    p.add_argument(
        '--transformer_layers', type=int, default=None,
        help='Optional architecture check; checkpoint value is used by default',
    )
    p.add_argument('--output', type=str, default='./recon.nii.gz')
    p.add_argument('--output_dir', type=str, default='./outputs/')
    p.add_argument('--ct_range', type=float, nargs=2, default=None,
                   help='Optional HU override; checkpoint metadata is used by default')
    args = p.parse_args()
    inference(args)
