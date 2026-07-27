"""
推理脚本: 从稀疏投影直接重建 CT 体素。

流程:
  稀疏投影 + 角度编码 → CNN → Transformer → 双码本 → 渐进解码 → NIfTI

用法:
  python src/inference.py --checkpoint \
      'logs/thorax_fast_finalview=6_256_v7/best_model_finalview=6_v7.pth' \
      --data_root ~/autodl-tmp/thorax --case_id 2026-06-04_065713 --final_view 6 --output recon.nii.gz
"""

# 标准库负责路径、模块搜索路径和命令行参数。
import os, sys, argparse
# PyTorch 执行模型推理，NumPy 用于 NIfTI 输出数组。
import torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SparseViewReconstruction
from src.dataset import ThoraxCTDataset
from src.view_protocol import (
    resolve_view_curriculum,
    uniform_view_indices as protocol_view_indices,
)


def uniform_view_indices(total_views, n_views, device):
    """复用训练协议并把确定性索引转换到指定设备。"""
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
    # 旧 checkpoint 缺少完整的预处理和码本初始化状态，拒绝猜测加载。
    if int(checkpoint.get('checkpoint_format', 0)) != 8:
        raise ValueError(
            'Unsupported legacy checkpoint. Expected checkpoint_format=8 '
            'with cross-batch K-means initialization state.'
        )
    required = (
        'model_state','model_config','ct_normalization',
        'source_view_policy','source_view_range',
        'eval_views','final_view','run_version',
        'resolved_view_schedule','sampling_protocol','base_views',
    )
    # 顶层字段用于恢复模型、验证视角协议和解释输出 HU。
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(
            f'Checkpoint is missing required metadata: {missing}'
        )
    required_model_config = (
        'n_decoder_ups', 'transformer_layers',
        'hf_codebook_size', 'mf_codebook_size',
        'kmeans_iters', 'kmeans_samples_per_code',
        'kmeans_init_batches',
        'dead_code_threshold', 'dead_code_check_interval',
        'dead_code_warmup_steps', 'proj_size', 'vol_size',
    )
    # 架构字段必须齐全，推理时不允许依靠当前源码默认值猜测。
    missing_model_config = [
        key for key in required_model_config
        if key not in checkpoint['model_config']
    ]
    if missing_model_config:
        raise ValueError(
            'Checkpoint model_config is missing required metadata: '
            f'{missing_model_config}'
        )
    # 只接受当前实现能够精确复现的源网格嵌套采样协议。
    expected_protocol = (
        'uniform_max_base_then_random_train_subset_fixed_eval'
    )
    if checkpoint['sampling_protocol'] != expected_protocol:
        raise ValueError(
            f'Unsupported sampling_protocol='
            f'{checkpoint["sampling_protocol"]!r}; expected '
            f'{expected_protocol!r}'
        )
    return checkpoint


def load_model(checkpoint_path, device, n_decoder_ups=None,
               transformer_layers=None,
               expected_views=None,
               checkpoint_data=None):
    # 调用方已预读 checkpoint 时直接复用，避免从磁盘重复加载。
    ckpt = (
        checkpoint_data
        if checkpoint_data is not None
        else torch.load(checkpoint_path, map_location='cpu')
    )
    validate_checkpoint_metadata(ckpt)
    model_config = ckpt['model_config']
    # 模型始终以 checkpoint 架构为准；命令行参数仅用于一致性校验。
    saved_ups = int(model_config['n_decoder_ups'])
    saved_transformer_layers = int(model_config['transformer_layers'])
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
    # 按保存时的全部结构参数重建网络后再加载 state_dict。
    model = SparseViewReconstruction(
        n_decoder_ups=saved_ups,
        transformer_layers=saved_transformer_layers,
        hf_codebook_size=int(model_config['hf_codebook_size']),
        mf_codebook_size=int(model_config['mf_codebook_size']),
        kmeans_iters=int(model_config['kmeans_iters']),
        kmeans_samples_per_code=int(
            model_config['kmeans_samples_per_code']
        ),
        kmeans_init_batches=int(model_config['kmeans_init_batches']),
        dead_code_threshold=float(model_config['dead_code_threshold']),
        dead_code_check_interval=int(
            model_config['dead_code_check_interval']
        ),
        dead_code_warmup_steps=int(
            model_config['dead_code_warmup_steps']
        ),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    # 推理模式关闭 dropout、BatchNorm 更新和 EMA 码本更新。
    model.eval()
    metric=ckpt.get('best_val_psnr','N/A')
    eval_views=ckpt['eval_views']
    source_view_range=ckpt['source_view_range']
    run_version=ckpt['run_version']
    if expected_views is not None:
        if int(eval_views) != int(expected_views):
            raise ValueError(
                f'Checkpoint was selected at {eval_views} views, '
                f'but inference requested {expected_views}'
            )
    print(
        f'Loaded epoch {ckpt.get("epoch","?")}, '
        f'Validation PSNR={metric}, eval_views={eval_views}, '
        f'per-case source views={source_view_range}, '
        f'run_version={run_version}, '
        f'transformer_layers={saved_transformer_layers}, '
        f'codebooks=HF{model_config["hf_codebook_size"]}/'
        f'MF{model_config["mf_codebook_size"]}'
    )
    return model,ckpt


@torch.no_grad()
def reconstruct_single(
    model, dataset, case_id, device, n_views=6,
    ct_range=(-1000, 1000), base_views=None,
):
    # case_id 必须属于 Dataset 当前 split。
    try:
        idx = dataset.cases.index(case_id)
    except ValueError:
        raise ValueError(f'Case "{case_id}" not found.')

    # 当前 Dataset 会加载完整源视角以及 CT；模型输入只使用投影和角度。
    batch = dataset[idx]
    source_projs = batch['projs'][:, 0, :, :]
    source_angles = batch['angles']
    source_views = int(batch['source_views'])
    base_views = n_views if base_views is None else int(base_views)
    # 第一步从病例实际源网格建立固定最大基准集。
    base_indices = uniform_view_indices(
        source_views,base_views,batch['view_indices'].device
    )
    # 第二步从基准集取得最终厂家部署子集。
    within_base = uniform_view_indices(
        base_views,n_views,batch['view_indices'].device
    )
    expected_indices = base_indices.index_select(0, within_base)
    # 将目标原始索引映射为当前已加载投影张量中的位置。
    matches = batch['view_indices'][:, None].eq(expected_indices[None, :])
    if not torch.all(matches.sum(dim=0) == 1):
        raise ValueError(
            f'Case {case_id!r} does not contain the expected direct '
            f'{source_views}->{base_views}->{n_views} source-grid protocol'
        )
    positions = matches.to(torch.int64).argmax(dim=0)
    projs = source_projs.index_select(0, positions)
    angles = source_angles.index_select(0, positions)
    projs_enc = add_angle_encoding(projs.to(device),angles,device)   # (V, 3, H, W)
    projs_enc = projs_enc.unsqueeze(0)                               # (1, V, 3, H, W)

    # 单病例增加 batch 维后执行完整重建前向。
    pred, _, _ = model(projs_enc)
    ct_min, ct_max = ct_range
    normalized = pred[0, 0].float()
    # 线性输出头不保证 [0,1]，保存前统计并报告越界比例。
    outside_fraction = float(
        ((normalized < 0.0) | (normalized > 1.0)).float().mean()
    )
    if outside_fraction > 0:
        print(
            f'Warning: {outside_fraction:.2%} predicted voxels were outside '
            '[0,1] and were clipped before HU conversion.'
        )
    # 裁剪到训练标签定义域后，按 checkpoint HU 窗口线性反归一化。
    volume = normalized.clamp(0.0, 1.0).cpu().numpy()
    volume = volume * (ct_max - ct_min) + ct_min
    return volume.astype(np.float32)


def resampled_affine(reference_affine, reference_shape, output_shape):
    """Preserve reference orientation/FOV for a differently sized output grid."""
    reference_shape = np.asarray(reference_shape[:3], dtype=np.float64)
    output_shape = np.asarray(output_shape[:3], dtype=np.float64)
    # 保持相同物理 FOV 时，输出矩阵变大意味着每个 voxel spacing 变小。
    scale = reference_shape / output_shape
    voxel_transform = np.eye(4, dtype=np.float64)
    voxel_transform[0,0],voxel_transform[1,1],voxel_transform[2,2] = scale
    # align_corners=False-style center alignment
    voxel_transform[:3,3] = 0.5 * scale - 0.5
    return np.asarray(reference_affine) @ voxel_transform


def save_nifti(volume, path, reference_path):
    """按参考 CT 的方向与物理视野保存 float32 HU NIfTI。"""
    import nibabel as nib

    # 模型使用 (D,H,W)，nibabel 数组使用 (X,Y,Z)，因此恢复轴顺序。
    vol = np.transpose(volume, (2, 1, 0))
    reference = nib.load(reference_path)
    affine = resampled_affine(
        reference.affine,reference.shape,vol.shape
    )
    # 复制医学空间元数据，但覆盖数据类型、缩放和显示范围。
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
    """校验 checkpoint/协议，逐病例重建并保存 NIfTI。"""
    # 自动选择 CUDA；无 GPU 时仍可执行但三维模型会很慢。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.source_views == 0 or args.source_views < -1:
        raise ValueError('source_views must be -1 or a positive integer')
    # 命令行协议必须和 checkpoint 保存的训练协议完全一致。
    requested_schedule = resolve_view_curriculum(
        args.final_view,args.view_schedule,max_views=64
    )
    n_views=args.final_view
    # 先在 CPU 上读取元数据，完成校验后才创建 GPU 模型。
    checkpoint_preview=validate_checkpoint_metadata(
        torch.load(args.checkpoint,map_location='cpu')
    )
    saved_schedule = tuple(
        int(value)
        for value in checkpoint_preview['resolved_view_schedule']
    )
    if tuple(requested_schedule) != saved_schedule:
        raise ValueError(
            f'Inference view schedule {list(requested_schedule)} does not '
            f'match checkpoint schedule {list(saved_schedule)}'
        )
    base_views = int(checkpoint_preview['base_views'])
    if base_views != max(saved_schedule):
        raise ValueError(
            f'Checkpoint base_views={base_views} does not match the maximum '
            f'saved schedule view count {max(saved_schedule)}'
        )
    model_config=checkpoint_preview['model_config']
    # 投影预处理尺寸和 CT 元数据都由 checkpoint 决定。
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
    # 结构校验通过后加载权重到最终推理设备。
    model,_ = load_model(
        args.checkpoint,device,n_decoder_ups=args.n_decoder_ups,
        transformer_layers=args.transformer_layers,
        expected_views=n_views,
        checkpoint_data=checkpoint_preview,
    )
    dataset = ThoraxCTDataset(data_root=args.data_root,
                             split=args.split if args.case_id is None else 'test',
                             n_views=-1,
                             proj_size=saved_proj_size,
                             vol_size=saved_vol_size,
                             expected_source_views=(
                                 None
                                 if args.source_views == -1
                                 else args.source_views
                             ))
    print(
        f'Dataset: {len(dataset)} cases, '
        f'per-case source views={dataset.min_views}..{dataset.max_views} '
        f'-> uniform {base_views}-view base -> fixed '
        f'{n_views}-view inference'
    )

    if args.case_id:
        # 指定病例时写入单个 --output 文件。
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        vol = reconstruct_single(
            model,dataset,args.case_id,device,n_views,ct_range,base_views
        )
        save_nifti(vol,args.output,dataset.ct_path(args.case_id))
    else:
        # 未指定病例时遍历 split，并按病例 ID 写入输出目录。
        os.makedirs(args.output_dir, exist_ok=True)
        for cid in dataset.cases:
            print(f'Processing: {cid}')
            vol = reconstruct_single(
                model,dataset,cid,device,n_views,ct_range,base_views
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
    p.add_argument(
        '--source_views', type=int, default=-1,
        help='-1 loads each case in full; a positive value optionally asserts '
             'an identical source-view count',
    )
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
