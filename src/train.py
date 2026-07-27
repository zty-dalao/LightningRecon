"""
训练脚本: 固定等间隔厂家协议 + 三阶段训练 + Charbonnier 主损失

阶段1 (Stage1): 60/64 views, EMA码字学习, L_img 主导
阶段2 (Stage2): 内置课程逐级降至目标视角, EMA码字冻结
阶段3 (Stage3): 从基准集随机抽取目标视角数, EMA码字和量化适配层冻结

核心改动:
  - L_img (Charbonnier, 完整体积) 为主损失 (w=1.0)
  - 投影与真实物理角度始终绑定
  - 先从每个病例原始网格均匀建立60/64-view基准集
  - 训练中的低视角阶段从基准集随机无放回抽样；val/test使用固定协议
  - val 始终使用 --final_view 选择 checkpoint；test 仅在结束时评估
  - best/periodic/last checkpoint 均可完整恢复训练
  - Warmup + CosineAnnealingLR, 每阶段重置
  - 分组学习率 (encoder / codebook / decoder)

用法:
  python src/train.py --data_root /root/autodl-tmp/thorax \
      --stage1_epochs 150 --stage2_epochs_per_view 30 --stage3_epochs 100
"""

# 标准库分别负责路径、命令行、配置序列化、版本名校验和随机状态。
import os, sys, argparse, json, re, random
# NumPy 用于 RNG 状态与少量统计；PyTorch 承担模型训练。
import numpy as np
import torch, torch.nn as nn
# DataLoader 负责并行病例读取；default_collate 处理固定尺寸字段。
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import SparseViewReconstruction
from src.dataset import ThoraxCTDataset
from src.losses import ReconstructionLoss, ssim_3d_per_case
from src.view_protocol import (
    resolve_view_curriculum,
    uniform_view_indices as protocol_view_indices,
)


# =========================================================================
# 工具
# =========================================================================

def validate_run_version(run_version):
    """校验日志版本名必须是 v1、v2、v3 等显式正整数版本。"""
    if not re.fullmatch(r'v[1-9]\d*',run_version or ''):
        raise ValueError(
            f'run_version must look like v1, v2, v3, ...; got {run_version!r}'
        )
    return run_version


def build_run_name(organ, final_view, out_res, run_version):
    """把器官、最终视角、输出分辨率和版本组合成唯一运行目录名。"""
    version=validate_run_version(run_version)
    return f'{organ}_finalview={final_view}_{out_res}_{version}'


def build_tensorboard_command(tensorboard_dir):
    """生成只查看当前版本日志目录的 TensorBoard 命令。"""
    return f'tensorboard --logdir "{os.path.abspath(tensorboard_dir)}"'


def checkpoint_name(kind, final_view, run_version, epoch=None):
    """为 best、last 或周期 checkpoint 生成可追溯文件名。"""
    version=validate_run_version(run_version)
    suffix=f'finalview={final_view}_{version}'
    if kind=='best':
        return f'best_model_{suffix}.pth'
    if kind=='last':
        return f'last_model_{suffix}.pth'
    if kind=='epoch' and epoch is not None:
        return f'ckpt_{epoch:04d}_{suffix}.pth'
    raise ValueError(f'Unsupported checkpoint kind={kind!r}, epoch={epoch!r}')


def add_angle_encoding(projs, angles):
    """Append sin/cos maps from the physical acquisition angles (radians)."""
    # 投影本身占一个通道，后面将增加 sin(theta) 与 cos(theta) 两个通道。
    B, V, _, H, W = projs.shape
    if angles.dim() == 1:
        angles = angles.unsqueeze(0)
    if angles.shape != (B, V):
        raise ValueError(
            f'Expected angles shape {(B, V)}, got {tuple(angles.shape)}'
        )
    # 角度随投影移动到同一设备和精度，避免隐式复制。
    theta = angles.to(device=projs.device, dtype=projs.dtype)
    s = torch.sin(theta).view(B, V, 1, 1, 1).expand(-1, -1, 1, H, W)
    c = torch.cos(theta).view(B, V, 1, 1, 1).expand(-1, -1, 1, H, W)
    return torch.cat([projs, s, c], dim=2)


def uniform_view_indices(total_views, n_keep, device):
    """Fixed approximately uniform indices, preserving acquisition phase."""
    # 真实索引计算集中在 view_protocol，当前函数只负责转成 Tensor。
    indices=protocol_view_indices(total_views,n_keep)
    return torch.tensor(indices,device=device,dtype=torch.long)


def collate_variable_projection_batch(samples):
    """Keep full projection sequences as lists; collate fixed-size fields."""
    # 不同病例可能有 316、464、491 等不同源视角数，不能直接 stack。
    variable_keys = {'projs', 'angles', 'view_indices'}
    batch = {
        key: [sample[key] for sample in samples]
        for key in variable_keys
    }
    # CT、case_id 和 source_views 等固定结构字段仍使用 PyTorch 默认拼接。
    for key in samples[0]:
        if key not in variable_keys:
            batch[key] = default_collate([sample[key] for sample in samples])
    return batch


def subsample_projections(
    projs, n_keep, device, angles=None, *,
    source_total=None, view_indices=None, base_views=None,
    random_subset=False,
):
    """Select paired projections/angles through a uniform maximum-view base."""
    # 所有低视角子集先建立在同一最大基准集上。
    base_views = n_keep if base_views is None else int(base_views)
    if n_keep <= 0 or base_views < n_keep:
        raise ValueError(
            f'Expected 0 < n_keep <= base_views, got '
            f'{n_keep} and {base_views}'
        )
    # 可变长度 batch 逐病例采样，最后得到相同 n_keep 后再 stack。
    if isinstance(projs, (list, tuple)):
        if angles is not None and not isinstance(angles, (list, tuple)):
            raise ValueError('Variable projection batches require angle lists')
        if view_indices is not None and not isinstance(
            view_indices, (list, tuple)
        ):
            raise ValueError(
                'Variable projection batches require view-index lists'
            )
        if isinstance(source_total, torch.Tensor):
            totals = source_total.detach().view(-1).tolist()
        elif isinstance(source_total, (tuple, list)):
            totals = list(source_total)
        else:
            totals = [source_total] * len(projs)
        if len(totals) != len(projs):
            raise ValueError(
                f'Expected {len(projs)} source totals, got {len(totals)}'
            )
        selected_projs = []
        selected_angles = []
        for sample_index, sample_projs in enumerate(projs):
            # 递归复用单病例张量路径，确保两条路径采用完全相同的协议。
            result = subsample_projections(
                sample_projs.unsqueeze(0),
                n_keep,
                device,
                None if angles is None else angles[sample_index].unsqueeze(0),
                source_total=totals[sample_index],
                view_indices=(
                    None
                    if view_indices is None
                    else view_indices[sample_index].unsqueeze(0)
                ),
                base_views=base_views,
                random_subset=random_subset,
            )
            if angles is None:
                selected_projs.append(result[0])
            else:
                sample_selected, sample_angles = result
                selected_projs.append(sample_selected[0])
                selected_angles.append(sample_angles[0])
        projs_out = torch.stack(selected_projs).contiguous()
        if angles is None:
            return projs_out
        return projs_out, torch.stack(selected_angles).contiguous()

    if projs.dim() != 5:
        raise ValueError(f'Expected 5D, got {tuple(projs.shape)}')
    # 固定长度路径的第二维就是当前张量实际包含的视角数。
    V_total = projs.shape[1]
    if base_views > V_total:
        raise ValueError(
            f'base_views={base_views} exceeds loaded views={V_total}'
        )
    if view_indices is None:
        # 没有原始索引时，假定当前张量本身就是完整连续源网格。
        base_idx = uniform_view_indices(V_total, base_views, device)
        if n_keep == base_views:
            within_base = torch.arange(base_views, device=device)
        elif random_subset:
            # 训练阶段在基准集内无放回随机抽取，再排序恢复采集顺序。
            within_base = torch.randperm(
                base_views, device=device
            )[:n_keep].sort().values
        else:
            # 验证、测试和部署使用确定性的嵌套等间隔子集。
            within_base = uniform_view_indices(
                base_views, n_keep, device
            )
        idx = base_idx.index_select(0, within_base)
        sampled_projs = projs.index_select(1, idx).contiguous()
        if angles is None:
            return sampled_projs
        return sampled_projs, angles.index_select(1, idx).contiguous()

    # 有 view_indices 时，必须知道每个病例完整源网格的总视角数。
    if source_total is None:
        raise ValueError('source_total is required with view_indices')
    if view_indices.dim() == 1:
        view_indices = view_indices.unsqueeze(0)
    if view_indices.dim() != 2 or view_indices.shape != projs.shape[:2]:
        raise ValueError(
            f'Expected view_indices shape {tuple(projs.shape[:2])}, got '
            f'{tuple(view_indices.shape)}'
        )
    if isinstance(source_total, torch.Tensor):
        totals = source_total.detach().view(-1).tolist()
    elif isinstance(source_total, (tuple, list)):
        totals = list(source_total)
    else:
        totals = [source_total] * projs.shape[0]
    if len(totals) != projs.shape[0]:
        raise ValueError(
            f'Expected {projs.shape[0]} source totals, got {len(totals)}'
        )

    sampled_projs = []
    sampled_angles = []
    for sample_index, total in enumerate(totals):
        total = int(total)
        if base_views > total:
            raise ValueError(
                f'Batch sample {sample_index}: base_views={base_views} '
                f'exceeds source views={total}'
            )
        # available 将当前已加载位置映射回该病例的原始源网格。
        available = view_indices[sample_index].to(device)
        base_target = uniform_view_indices(total, base_views, device)
        if n_keep == base_views:
            within_base = torch.arange(base_views, device=device)
        elif random_subset:
            within_base = torch.randperm(
                base_views, device=device
            )[:n_keep].sort().values
        else:
            within_base = uniform_view_indices(
                base_views, n_keep, device
            )
        target = base_target.index_select(0, within_base)
        # 查找每个目标原始索引在当前投影张量中的唯一位置。
        matches = available[:, None].eq(target[None, :])
        if not torch.all(matches.sum(dim=0) == 1):
            missing = target[matches.sum(dim=0) != 1].tolist()
            raise ValueError(
                f'Batch sample {sample_index}: loaded projection union does '
                f'not contain the {total}->{base_views}->{n_keep} protocol; '
                f'missing indices={missing}'
            )
        positions = matches.to(torch.int64).argmax(dim=0)
        sampled_projs.append(
            projs[sample_index].index_select(0, positions)
        )
        if angles is not None:
            sampled_angles.append(
                angles[sample_index].index_select(0, positions)
            )
    projs_out = torch.stack(sampled_projs).contiguous()
    if angles is None:
        return projs_out
    return projs_out, torch.stack(sampled_angles).contiguous()


def _psnr(p, t):
    """按 data_range=1 计算单病例完整体积 PSNR。"""
    mse = torch.mean((p - t) ** 2)
    return float('inf') if mse == 0 else 20 * torch.log10(1.0 / torch.sqrt(mse))


# =========================================================================
# 视角调度
# =========================================================================

def build_view_schedule(args):
    """返回 [(start_epoch, end_epoch, n_views, stage), ...]"""
    # 先解析高到低的视角序列，再把各视角展开为闭区间 epoch 段。
    views=resolve_view_curriculum(
        args.final_view,args.view_schedule,max_views=64
    )
    schedule=[]
    s1_end = args.stage1_epochs
    # Stage 1 只使用课程最大视角并允许 EMA 更新。
    schedule.append((1,s1_end,views[0],1))

    ep_per_view=args.stage2_epochs_per_view
    cur=s1_end+1
    # Stage 2 依次遍历剩余视角数，包括 final_view。
    for nv in views[1:]:
        schedule.append((cur,cur+ep_per_view-1,nv,2))
        cur+=ep_per_view

    # Stage 3 可关闭；启用时在 final_view 上冻结量化适配层继续微调。
    if args.stage3_epochs > 0:
        s2_end = schedule[-1][1] if len(schedule) > 1 else s1_end
        schedule.append((
            s2_end+1,s2_end+args.stage3_epochs,args.final_view,3
        ))

    return schedule


def get_stage_config(stage):
    """(loss weights, learning rates, freeze EMA, freeze quantizer adapters)."""
    # 返回顺序：损失权重、分组学习率、是否冻结 EMA、是否冻结量化适配层。
    if stage == 1:
        return ({'w_img': 1.0, 'w_lap': 0.05, 'w_struct': 0.05, 'w_vq': 0.05},
                {'encoder': 1e-4, 'codebook': 1e-4, 'decoder': 1e-4},
                False, False)
    elif stage == 2:
        return ({'w_img': 1.0, 'w_lap': 0.04, 'w_struct': 0.08, 'w_vq': 0.02},
                {'encoder': 5e-5, 'codebook': 5e-6, 'decoder': 5e-5},
                True, False)
    else:
        return ({'w_img': 1.0, 'w_lap': 0.02, 'w_struct': 0.05, 'w_vq': 0.0},
                {'encoder': 1e-5, 'decoder': 2e-5},
                True, True)


def configure_stage(model, criterion, stage, view_schedule):
    """Apply freezing/loss policy and create the stage optimizer/scheduler."""
    loss_weights, lr_cfg, freeze_ema, freeze_cb_adapters = get_stage_config(stage)
    # 阶段切换时同步应用损失权重和冻结策略。
    criterion.set_weights(**loss_weights)
    if freeze_ema:
        model.freeze_codebooks()
    else:
        model.unfreeze_codebooks()
    model.set_codebook_adapters_trainable(not freeze_cb_adapters)

    # 通过参数对象 id 去重，确保每个参数只进入一个优化器分组。
    decoder_ids = {id(parameter) for parameter in model.decoder.parameters()}
    codebook_modules = [model.codebook_hf, model.codebook_mf]
    codebook_ids = {
        id(parameter)
        for codebook in codebook_modules
        for parameter in codebook.parameters()
    }
    encoder_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and id(parameter) not in codebook_ids
        and id(parameter) not in decoder_ids
    ]
    codebook_parameters = [
        parameter
        for codebook in codebook_modules
        for parameter in codebook.parameters()
        if parameter.requires_grad
    ]
    decoder_parameters = [
        parameter
        for parameter in model.decoder.parameters()
        if parameter.requires_grad
    ]
    parameter_groups = []
    if encoder_parameters:
        parameter_groups.append({
            'params': encoder_parameters,
            'lr': lr_cfg['encoder'],
            'name': 'encoder',
        })
    if codebook_parameters:
        parameter_groups.append({
            'params': codebook_parameters,
            'lr': lr_cfg.get('codebook', lr_cfg['encoder']),
            'name': 'codebook',
        })
    if decoder_parameters:
        parameter_groups.append({
            'params': decoder_parameters,
            'lr': lr_cfg.get('decoder', lr_cfg['encoder']),
            'name': 'decoder',
        })
    # 所有分组共享 AdamW 与 weight decay，但使用各自学习率。
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-5)

    # 同一 Stage 可能包含多个视角区间，scheduler 覆盖整个 Stage 总长度。
    stage_start = min(
        start for start, _, _, item_stage in view_schedule
        if item_stage == stage
    )
    stage_end = max(
        end for _, end, _, item_stage in view_schedule
        if item_stage == stage
    )
    stage_epochs = stage_end - stage_start + 1
    warmup_epochs = min(5, max(1, stage_epochs // 4))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_epochs
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, stage_epochs - warmup_epochs)
            ),
        ],
        milestones=[warmup_epochs],
    )
    return optimizer, scheduler, loss_weights, freeze_ema, freeze_cb_adapters


def capture_rng_state():
    """捕获 Python、NumPy、CPU Torch 与全部 CUDA 设备的随机状态。"""
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    """从 checkpoint 恢复随机状态，使随机视角课程能够连续执行。"""
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(
            [cuda_state.cpu() for cuda_state in state['cuda']]
        )


def checkpoint_payload(
    *,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    args,
    run_name,
    stage,
    current_train_views,
    resolved_views,
    best_metric,
    best_epoch,
    eval_metrics=None,
):
    """Build a fully resumable, self-describing training checkpoint."""
    # checkpoint 同时保存权重、优化器、调度器、AMP、RNG 和完整协议元数据。
    return {
        'checkpoint_format': 8,
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'scaler_state': scaler.state_dict() if scaler is not None else None,
        'rng_state': capture_rng_state(),
        'run_name': run_name,
        'run_version': args.run_version,
        'final_view': args.final_view,
        'best_val_psnr': best_metric,
        'best_epoch': best_epoch,
        'stage': stage,
        'current_train_views': current_train_views,
        'eval_views': args.final_view,
        'source_view_policy': 'per_case_actual',
        'source_view_range': list(args.source_view_range),
        'resolved_view_schedule': resolved_views,
        'sampling_protocol': (
            'uniform_max_base_then_random_train_subset_fixed_eval'
        ),
        'base_views': max(resolved_views),
        'model_config': {
            'n_decoder_ups': args.n_decoder_ups,
            'transformer_layers': args.transformer_layers,
            'hf_codebook_size': args.hf_codebook_size,
            'mf_codebook_size': args.mf_codebook_size,
            'kmeans_iters': args.kmeans_iters,
            'kmeans_samples_per_code': args.kmeans_samples_per_code,
            'kmeans_init_batches': args.kmeans_init_batches,
            'dead_code_threshold': args.dead_code_threshold,
            'dead_code_check_interval': args.dead_code_check_interval,
            'dead_code_warmup_steps': args.dead_code_warmup_steps,
            'proj_size': list(args.proj_size),
            'vol_size': list(args.vol_size),
        },
        'training_config': {
            'stage1_epochs': args.stage1_epochs,
            'stage2_epochs_per_view': args.stage2_epochs_per_view,
            'stage3_epochs': args.stage3_epochs,
            'batch_size': args.batch_size,
            'grad_accum': args.grad_accum,
            'amp': scaler is not None,
            'codebook_feature_sampling': 'uniform',
        },
        'ct_normalization': {
            'stored_range': [0.0, 255.0],
            'network_range': [0.0, 1.0],
            'hu_range': list(args.ct_range),
        },
        'eval_metrics': eval_metrics,
    }


def load_training_checkpoint(path, device):
    """从指定设备映射加载完整训练 checkpoint。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Resume checkpoint not found: {path}')
    return torch.load(path, map_location=device)


@torch.no_grad()
def initialize_codebooks(
    model, loader, device, base_views, use_amp, pin_memory,
):
    """Collect fixed-network features across batches before optimization."""
    # resume 后若两个码本都已完成初始化，不重复扫描训练数据。
    if (
        bool(model.codebook_hf.codebook.kmeans_initialized.item())
        and bool(model.codebook_mf.codebook.kmeans_initialized.item())
    ):
        return
    print('[Codebooks] Collecting cross-batch K-means initialization features...')
    model.train()
    ready = False
    # 当训练病例少于 kmeans_init_batches 时允许循环多遍 loader 收集到足量 batch。
    while not ready:
        made_progress = False
        for batch in loader:
            made_progress = True
            # 初始化始终使用课程最大基准视角，且不做随机低视角抽样。
            projs, angles = subsample_projections(
                batch['projs'],base_views,torch.device('cpu'),
                batch['angles'],
                source_total=batch['source_views'],
                view_indices=batch['view_indices'],
                base_views=base_views,
                random_subset=False,
            )
            projs = projs.to(device,non_blocking=pin_memory)
            angles = angles.to(device,non_blocking=pin_memory)
            projs_encoded = add_angle_encoding(projs,angles)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    ready = model.collect_codebook_initialization(
                        projs_encoded
                    )
            else:
                ready = model.collect_codebook_initialization(projs_encoded)
            progress = float(
                model.codebook_hf.codebook.diagnostics().get(
                    'kmeans_init_progress',torch.tensor(0.0)
                )
            )
            print(f'  K-means feature collection: {progress:.0%}')
            if ready:
                break
        if not made_progress:
            raise RuntimeError(
                'Cannot initialize codebooks from an empty train loader'
            )
    print('[Codebooks] Cross-batch K-means initialization complete.')


# =========================================================================
# 验证
# =========================================================================

@torch.no_grad()
def evaluate(
    model, loader, device, criterion, n_views, source_total=None,
    base_views=None,
):
    """Evaluate a fixed deployment protocol at exactly ``n_views`` views."""
    # eval 模式会停止 dropout、BN 统计和 EMA 码本更新。
    model.eval()
    m = {
        'psnr':0,'ssim':0,'total':0,'img':0,'lap':0,'struct':0,
        'perplexity':0,'c':0,
    }
    for batch in loader:
        # 每个病例先从其完整源网格提取固定部署子集。
        source_projs=batch['projs']
        source_angles=batch['angles']
        ct=batch['ct'].to(device)
        view_indices=batch.get('view_indices')
        case_source_totals = batch.get('source_views', source_total)
        projs,angles=subsample_projections(
            source_projs,n_views,torch.device('cpu'),source_angles,
            source_total=case_source_totals,
            view_indices=view_indices,
            base_views=base_views,
            random_subset=False,
        )
        projs=projs.to(device)
        angles=angles.to(device)
        projs_enc=add_angle_encoding(projs,angles)
        # 验证前向不建立梯度图。
        pred,vq,perplexity=model(projs_enc)
        ct_a=nn.functional.interpolate(ct, size=pred.shape[2:], mode='trilinear')
        loss=criterion(pred,ct_a,vq,compute_metrics=False)
        B=projs.shape[0]
        whole_ssim_scores=ssim_3d_per_case(pred,ct_a)
        # PSNR 必须逐病例计算后平均，不能先跨 batch 合并 MSE。
        for sample_index in range(B):
            m['psnr']+=float(_psnr(
                pred[sample_index:sample_index+1],
                ct_a[sample_index:sample_index+1],
            ))
        m['ssim']+=float(whole_ssim_scores.sum())
        for k in ['total','img','lap','struct']:
            v=loss[k]; m[k]+=(v.item() if isinstance(v, torch.Tensor) else v)*B
        m['perplexity']+=float(perplexity)*B
        m['c']+=B
    # 所有返回指标均为病例级平均值。
    return {'psnr':m['psnr']/m['c'],
            'ssim':m['ssim']/m['c'], 'total':m['total']/m['c'],
            'img':m['img']/m['c'], 'lap':m['lap']/m['c'],
            'struct':m['struct']/m['c'],
            'perplexity':m['perplexity']/m['c']}


# =========================================================================
# 主训练
# =========================================================================

def train(args):
    """执行数据检查、三阶段训练、验证选模和最终独立测试。"""
    # 有 CUDA 时自动使用 GPU，否则退回 CPU 便于小规模逻辑测试。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 课程一经解析便写入 checkpoint，resume 时必须完全一致。
    view_schedule = build_view_schedule(args)
    resolved_views = [nv for _, _, nv, stage in view_schedule if stage != 3]
    if args.eval_every <= 0 or args.save_every <= 0:
        raise ValueError('eval_every and save_every must be positive')
    if args.source_views == 0 or args.source_views < -1:
        raise ValueError('source_views must be -1 or a positive integer')
    if args.transformer_layers <= 0:
        raise ValueError('transformer_layers must be positive')
    if args.hf_codebook_size <= 0 or args.mf_codebook_size <= 0:
        raise ValueError('HF and MF codebook sizes must be positive')
    if args.kmeans_iters <= 0 or args.kmeans_samples_per_code < 1:
        raise ValueError(
            'kmeans_iters must be positive and '
            'kmeans_samples_per_code must be at least 1'
        )
    if args.kmeans_init_batches < 1:
        raise ValueError('kmeans_init_batches must be at least 1')
    if args.dead_code_threshold < 0:
        raise ValueError('dead_code_threshold must be non-negative')
    if args.dead_code_check_interval <= 0:
        raise ValueError('dead_code_check_interval must be positive')
    if args.dead_code_warmup_steps < 0:
        raise ValueError('dead_code_warmup_steps must be non-negative')
    if args.ct_range[0] >= args.ct_range[1]:
        raise ValueError(f'ct_range must be increasing, got {args.ct_range}')
    # 最后一个课程区间的结束 epoch 就是完整训练轮数。
    total_epochs = view_schedule[-1][1]
    organ = getattr(args, 'organ', 'thorax_fast')
    out_res = 128 * (2 ** args.n_decoder_ups)
    run_name = build_run_name(
        organ, args.final_view, out_res, args.run_version
    )
    log_dir = os.path.join(args.log_dir, run_name)
    # 在创建数据集和模型前先读取 checkpoint 元数据，尽早拒绝不兼容配置。
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = load_training_checkpoint(args.resume, device)
        if int(resume_checkpoint.get('checkpoint_format', 0)) != 8:
            raise ValueError(
                'This checkpoint is not compatible with checkpoint_format=8. '
                'Start a new run because cross-batch K-means initialization '
                'state was added.'
            )
        for key, expected in (
            ('run_version', args.run_version),
            ('final_view', args.final_view),
        ):
            actual = resume_checkpoint.get(key)
            if actual is not None and actual != expected:
                raise ValueError(
                    f'Resume mismatch for {key}: checkpoint={actual!r}, '
                    f'command line={expected!r}'
                )
        if resume_checkpoint.get('run_name') not in (None, run_name):
            raise ValueError(
                f'Resume checkpoint belongs to '
                f'{resume_checkpoint.get("run_name")!r}, expected {run_name!r}'
            )
        if resume_checkpoint.get('resolved_view_schedule') != resolved_views:
            raise ValueError(
                'Resume view schedule differs from the checkpoint: '
                f'{resume_checkpoint.get("resolved_view_schedule")} vs '
                f'{resolved_views}'
            )
        # 架构和预处理配置不同会导致权重形状或输入语义不一致。
        expected_model_config = {
            'n_decoder_ups': args.n_decoder_ups,
            'transformer_layers': args.transformer_layers,
            'hf_codebook_size': args.hf_codebook_size,
            'mf_codebook_size': args.mf_codebook_size,
            'kmeans_iters': args.kmeans_iters,
            'kmeans_samples_per_code': args.kmeans_samples_per_code,
            'kmeans_init_batches': args.kmeans_init_batches,
            'dead_code_threshold': args.dead_code_threshold,
            'dead_code_check_interval': args.dead_code_check_interval,
            'dead_code_warmup_steps': args.dead_code_warmup_steps,
            'proj_size': list(args.proj_size),
            'vol_size': list(args.vol_size),
        }
        if resume_checkpoint.get('model_config') != expected_model_config:
            raise ValueError(
                'Resume model configuration differs from the checkpoint: '
                f'{resume_checkpoint.get("model_config")} vs '
                f'{expected_model_config}'
            )
        # 训练阶段长度和梯度设置也必须一致，保证 scheduler/optimizer 可恢复。
        expected_training_config = {
            'stage1_epochs': args.stage1_epochs,
            'stage2_epochs_per_view': args.stage2_epochs_per_view,
            'stage3_epochs': args.stage3_epochs,
            'batch_size': args.batch_size,
            'grad_accum': args.grad_accum,
            'amp': bool(args.amp and device.type == 'cuda'),
            'codebook_feature_sampling': 'uniform',
        }
        if resume_checkpoint.get('training_config') != expected_training_config:
            raise ValueError(
                'Resume training configuration differs from the checkpoint: '
                f'{resume_checkpoint.get("training_config")} vs '
                f'{expected_training_config}'
            )
        saved_hu_range = (
            resume_checkpoint.get('ct_normalization', {}).get('hu_range')
        )
        if saved_hu_range != list(args.ct_range):
            raise ValueError(
                f'Resume ct_range differs from the checkpoint: '
                f'{saved_hu_range} vs {list(args.ct_range)}'
            )
    elif os.path.isdir(log_dir) and os.listdir(log_dir):
        raise FileExistsError(
            f'Run directory is not empty: {log_dir}. Use --resume to continue '
            'this run or choose a new --run_version.'
        )

    # Dataset 始终加载每个病例的全部投影；真正的课程采样在 batch 内完成。
    dataset_kwargs = {
        'data_root': args.data_root,
        'n_views': -1,
        'proj_size': tuple(args.proj_size),
        'vol_size': tuple(args.vol_size),
        'expected_source_views': (
            None if args.source_views == -1 else args.source_views
        ),
    }
    train_set = ThoraxCTDataset(split='train', **dataset_kwargs)
    val_set = ThoraxCTDataset(split='val', **dataset_kwargs)
    test_set = ThoraxCTDataset(split='test', **dataset_kwargs)
    # 汇总三个 split，确保课程最大视角数能被所有病例支持。
    all_source_counts = {
        **train_set.source_view_counts,
        **val_set.source_view_counts,
        **test_set.source_view_counts,
    }
    args.source_view_range = (
        min(all_source_counts.values()),
        max(all_source_counts.values()),
    )
    if args.source_view_range[0] < max(resolved_views):
        raise ValueError(
            f'At least one case has only {args.source_view_range[0]} source '
            f'views, below curriculum maximum {max(resolved_views)}'
        )
    if resume_checkpoint is not None:
        saved_policy = resume_checkpoint.get('source_view_policy')
        saved_range = resume_checkpoint.get('source_view_range')
        if saved_policy != 'per_case_actual':
            raise ValueError(
                f'Resume source-view policy is incompatible: {saved_policy!r}'
            )
        if saved_range != list(args.source_view_range):
            raise ValueError(
                f'Resume source-view range differs from the current dataset: '
                f'{saved_range} vs {list(args.source_view_range)}'
            )
    pin_memory = device.type == 'cuda'
    # 训练集打乱；验证和测试固定顺序，便于复现实验指标。
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory,
        collate_fn=collate_variable_projection_batch,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
        collate_fn=collate_variable_projection_batch,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
        collate_fn=collate_variable_projection_batch,
    )
    print(
        f'Train:{len(train_set)} Val:{len(val_set)} Test:{len(test_set)} | '
        f'per-case source views={args.source_view_range[0]}..'
        f'{args.source_view_range[1]}'
    )
    print(
        f'Protocol: full source grid -> uniform {max(resolved_views)}-view '
        f'base; each training batch randomly samples its requested view '
        f'count without replacement; val/test use the fixed '
        f'{args.final_view}-view subset'
    )
    print(f'View schedule ({len(view_schedule)} phases, {total_epochs} epochs):')
    for start, end, n_views, stage in view_schedule:
        print(
            f'  E{start:4d}-{end:4d} ({end-start+1:3d}ep): '
            f'{n_views:2d}v, Stage {stage}'
        )

    # 所有会改变 state_dict 形状或码本行为的参数都来自命令行。
    model = SparseViewReconstruction(
        n_decoder_ups=args.n_decoder_ups,
        transformer_layers=args.transformer_layers,
        hf_codebook_size=args.hf_codebook_size,
        mf_codebook_size=args.mf_codebook_size,
        kmeans_iters=args.kmeans_iters,
        kmeans_samples_per_code=args.kmeans_samples_per_code,
        kmeans_init_batches=args.kmeans_init_batches,
        dead_code_threshold=args.dead_code_threshold,
        dead_code_check_interval=args.dead_code_check_interval,
        dead_code_warmup_steps=args.dead_code_warmup_steps,
    ).to(device)
    criterion = ReconstructionLoss()
    use_amp = bool(args.amp and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    print(f'Model: {sum(p.numel() for p in model.parameters()):,} params')

    # 运行目录同时保存 TensorBoard、config、checkpoint 和最终测试指标。
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_dir = os.path.join(log_dir, 'tensorboard')
    tensorboard_cmd = build_tensorboard_command(tensorboard_dir)
    best_checkpoint = checkpoint_name(
        'best', args.final_view, args.run_version
    )
    last_checkpoint = checkpoint_name(
        'last', args.final_view, args.run_version
    )
    start_epoch = 1
    best_metric = -float('inf')
    best_epoch = 0
    if resume_checkpoint is not None:
        # 先恢复模型和全局训练进度；优化器要等当前 Stage 创建后再恢复。
        model.load_state_dict(resume_checkpoint['model_state'])
        start_epoch = int(resume_checkpoint['epoch']) + 1
        best_metric = float(
            resume_checkpoint.get('best_val_psnr', -float('inf'))
        )
        best_epoch = int(resume_checkpoint.get('best_epoch', 0))
        if start_epoch > total_epochs:
            print(
                f'Checkpoint already reaches epoch {start_epoch - 1}; '
                'skipping training and running the held-out test handoff.'
            )
        restore_rng_state(resume_checkpoint.get('rng_state'))
        print(f'Resuming from epoch {start_epoch - 1}: {args.resume}')

    # 新训练在第一次优化前完成跨 batch K-means；resume 通常会直接跳过。
    initialize_codebooks(
        model,train_loader,device,max(resolved_views),use_amp,pin_memory
    )

    # purge_step 避免 resume 后旧 event 中重叠 step 被 TensorBoard 重复展示。
    writer = SummaryWriter(
        tensorboard_dir,
        purge_step=start_epoch if resume_checkpoint is not None else None,
    )
    writer.add_text(
        'Run/Metadata',
        '\n'.join([
            f'- run_name: `{run_name}`',
            f'- run_version: `{args.run_version}`',
            f'- source_view_policy: `per_case_actual`',
            f'- source_view_range: `{args.source_view_range}`',
            f'- projection_loading: `all_views_per_case`',
            f'- training_view_sampling: `uniform '
            f'{max(resolved_views)}-view base, then random subset without '
            f'replacement per batch`',
            f'- eval_view_sampling: `fixed uniform '
            f'{args.final_view}-view subset of the base`',
            f'- final_view: `{args.final_view}`',
            f'- transformer_layers: `{args.transformer_layers}`',
            f'- codebook_sizes: `HF={args.hf_codebook_size}, '
            f'MF={args.mf_codebook_size}`',
            f'- codebook_initialization: `K-means on uniform features from '
            f'{args.kmeans_init_batches} shuffled training batches`',
            f'- dead_code_reinitialization: `enabled in EMA-update stage`',
            f'- codebook_feature_sampling: `uniform`',
            f'- utilization_review: `epochs 1-50`',
            f'- resolved_view_schedule: `{resolved_views}`',
            f'- tensorboard: `{tensorboard_cmd}`',
            f'- best_checkpoint: `{best_checkpoint}`',
            f'- last_checkpoint: `{last_checkpoint}`',
            f'- resumed_from: `{args.resume or "new run"}`',
        ]),
        max(0, start_epoch - 1),
    )
    # config.json 是人类可读的运行快照，不代替 checkpoint 中的恢复状态。
    config = vars(args).copy()
    config.update({
        'run_name': run_name,
        'train_cases': len(train_set),
        'val_cases': len(val_set),
        'test_cases': len(test_set),
        'total_epochs': total_epochs,
        'sampling_protocol': (
            'uniform_max_base_then_random_train_subset_fixed_eval'
        ),
        'base_views': max(resolved_views),
        'source_view_policy': 'per_case_actual',
        'source_view_range': list(args.source_view_range),
        'projection_loading': 'all_views_per_case',
        'codebook_feature_sampling': 'uniform',
        'codebook_utilization_review_epochs': [1, 50],
        'eval_views': args.final_view,
        'resolved_view_schedule': resolved_views,
        'tensorboard_dir': tensorboard_dir,
        'tensorboard_command': tensorboard_cmd,
        'best_checkpoint': best_checkpoint,
        'last_checkpoint': last_checkpoint,
    })
    with open(os.path.join(log_dir, 'config.json'), 'w') as handle:
        json.dump(config, handle, indent=2)
    print(f'TensorBoard for this run:\n  {tensorboard_cmd}')

    # 梯度累积用于在 batch_size=1 时扩大有效 batch。
    accumulation_steps = max(1, args.grad_accum)
    previous_stage = None
    optimizer = None
    scheduler = None
    resume_optimizer_pending = resume_checkpoint is not None
    last_epoch = (
        int(resume_checkpoint['epoch'])
        if resume_checkpoint is not None else 0
    )
    last_stage = (
        int(resume_checkpoint['stage'])
        if resume_checkpoint is not None else None
    )
    last_views = (
        int(resume_checkpoint['current_train_views'])
        if resume_checkpoint is not None else None
    )

    # epoch 使用从 1 开始的闭区间编号，与 checkpoint 文件名一致。
    for epoch in range(start_epoch, total_epochs + 1):
        current_stage = None
        current_views = None
        for start, end, n_views, stage in view_schedule:
            if start <= epoch <= end:
                current_stage, current_views = stage, n_views
                break
        if current_stage is None:
            raise RuntimeError(f'No curriculum phase covers epoch {epoch}')

        if current_stage != previous_stage:
            # 只有 Stage 改变时重建优化器和 scheduler；Stage 2 内各视角连续训练。
            (
                optimizer,
                scheduler,
                loss_weights,
                freeze_ema,
                freeze_cb_adapters,
            ) = configure_stage(
                model, criterion, current_stage, view_schedule
            )
            if resume_optimizer_pending:
                # resume checkpoint 属于当前 Stage 时才能安全恢复分组优化器状态。
                checkpoint_stage = int(resume_checkpoint['stage'])
                if checkpoint_stage == current_stage:
                    optimizer.load_state_dict(
                        resume_checkpoint['optimizer_state']
                    )
                    scheduler.load_state_dict(
                        resume_checkpoint['scheduler_state']
                    )
                    if scaler is not None and resume_checkpoint.get('scaler_state'):
                        scaler.load_state_dict(
                            resume_checkpoint['scaler_state']
                        )
                    print('Restored optimizer, scheduler, and AMP scaler state.')
                resume_optimizer_pending = False
            print(f'\n{"=" * 60}')
            print(
                f'[Stage {current_stage}] Epoch {epoch}: '
                f'views={current_views}'
            )
            print(
                f'  Loss: img={loss_weights["w_img"]} '
                f'lap={loss_weights["w_lap"]} '
                f'struct={loss_weights["w_struct"]} '
                f'vq={loss_weights["w_vq"]}'
            )
            print(
                f'  EMA codewords: {"frozen" if freeze_ema else "updating"} | '
                f'quantizer adapters: '
                f'{"frozen" if freeze_cb_adapters else "trainable"}'
            )
            previous_stage = current_stage

        # 重新进入训练模式，使 dropout、BN 和允许状态下的 EMA 正常更新。
        model.train()
        epoch_sums = {
            'total': 0.0, 'img': 0.0, 'lap': 0.0, 'struct': 0.0,
            'vq': 0.0, 'weighted_img': 0.0,
            'weighted_lap': 0.0, 'weighted_struct': 0.0,
            'weighted_vq': 0.0,
        }
        diagnostic_sums = {}
        diagnostic_last = {}
        # 每个 epoch 开始前清空上一轮可能残留的梯度。
        optimizer.zero_grad()
        for batch_index, batch in enumerate(train_loader):
            source_projs = batch['projs']
            source_angles = batch['angles']
            # 当前 Stage 的视角从统一最大基准集随机无放回抽取。
            projs, angles = subsample_projections(
                source_projs, current_views,
                torch.device('cpu'), source_angles,
                source_total=batch['source_views'],
                view_indices=batch['view_indices'],
                base_views=max(resolved_views),
                random_subset=True,
            )
            projs = projs.to(device, non_blocking=pin_memory)
            angles = angles.to(device, non_blocking=pin_memory)
            ct = batch['ct'].to(device, non_blocking=pin_memory)
            projs_encoded = add_angle_encoding(projs, angles)

            # AMP 只包围前向与损失；梯度裁剪前会先执行 unscale。
            if use_amp:
                with torch.amp.autocast('cuda'):
                    pred, vq, _ = model(projs_encoded)
                    ct_aligned = nn.functional.interpolate(
                        ct, size=pred.shape[2:], mode='trilinear',
                        align_corners=False,
                    )
                    losses = criterion(
                        pred,ct_aligned,vq,compute_metrics=False,
                    )
                scaler.scale(
                    losses['total'] / accumulation_steps
                ).backward()
            else:
                pred, vq, _ = model(projs_encoded)
                ct_aligned = nn.functional.interpolate(
                    ct, size=pred.shape[2:], mode='trilinear',
                    align_corners=False,
                )
                losses = criterion(
                    pred,ct_aligned,vq,compute_metrics=False,
                )
                (losses['total'] / accumulation_steps).backward()

            # 达到累积步数或 epoch 最后一个 batch 时执行一次参数更新。
            if (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index == len(train_loader) - 1
            ):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

            # 记录未加权损失、加权项和码本诊断，供 epoch 末统一平均。
            for name in epoch_sums:
                epoch_sums[name] += float(losses[name].detach())
            for name, value in model.codebook_diagnostics().items():
                scalar_value = float(value.detach())
                diagnostic_sums[name] = (
                    diagnostic_sums.get(name, 0.0)
                    + scalar_value
                )
                diagnostic_last[name] = scalar_value
        # scheduler 以 epoch 为单位推进一次。
        scheduler.step()

        num_batches = len(train_loader)
        averages = {
            name: value / num_batches
            for name, value in epoch_sums.items()
        }
        diagnostic_averages = {
            name: value / num_batches
            for name, value in diagnostic_sums.items()
        }
        for name in (
            'hf_dead_codes_reinitialized_total',
            'mf_dead_codes_reinitialized_total',
            'hf_kmeans_initialized',
            'mf_kmeans_initialized',
        ):
            if name in diagnostic_last:
                diagnostic_averages[name] = diagnostic_last[name]
        learning_rates = {
            group.get('name', f'group_{index}'): group['lr']
            for index, group in enumerate(optimizer.param_groups)
        }
        print(
            f'E{epoch:4d} S{current_stage} '
            f'lr={next(iter(learning_rates.values())):.2e} '
            f'V={current_views} | total={averages["total"]:.4f} '
            f'img={averages["img"]:.4f} lap={averages["lap"]:.4f} '
            f'struct={averages["struct"]:.4f} vq={averages["vq"]:.4f}'
        )
        if epoch <= 50 and diagnostic_averages:
            print(
                '  Codebook utilization | '
                f'HF active={diagnostic_averages.get("hf_ema_active_fraction", float("nan")):.3f} '
                f'norm_perp={diagnostic_averages.get("hf_normalized_perplexity", float("nan")):.3f} | '
                f'MF active={diagnostic_averages.get("mf_ema_active_fraction", float("nan")):.3f} '
                f'norm_perp={diagnostic_averages.get("mf_normalized_perplexity", float("nan")):.3f}'
            )
        for name, value in averages.items():
            writer.add_scalar(f'Train/Loss/{name}', value, epoch)
        weighted_total = sum(
            averages[name]
            for name in (
                'weighted_img','weighted_lap',
                'weighted_struct','weighted_vq',
            )
        )
        if weighted_total > 0:
            for name in (
                'weighted_img','weighted_lap',
                'weighted_struct','weighted_vq',
            ):
                writer.add_scalar(
                    f'Train/LossContribution/{name}',
                    averages[name] / weighted_total,
                    epoch,
                )
        for name, value in learning_rates.items():
            writer.add_scalar(f'Train/LearningRate/{name}', value, epoch)
        for name, value in diagnostic_averages.items():
            writer.add_scalar(f'Codebook/{name}', value, epoch)
        if epoch == 50:
            writer.add_text(
                'Codebook/Epoch50Review',
                '\n'.join([
                    '# Uniform-sampling utilization at epoch 50',
                    f'- HF EMA active fraction: '
                    f'`{diagnostic_averages.get("hf_ema_active_fraction", float("nan")):.4f}`',
                    f'- HF normalized perplexity: '
                    f'`{diagnostic_averages.get("hf_normalized_perplexity", float("nan")):.4f}`',
                    f'- MF EMA active fraction: '
                    f'`{diagnostic_averages.get("mf_ema_active_fraction", float("nan")):.4f}`',
                    f'- MF normalized perplexity: '
                    f'`{diagnostic_averages.get("mf_normalized_perplexity", float("nan")):.4f}`',
                    '- Sampling remains uniform; structure-aware sampling '
                    'is not enabled automatically.',
                ]),
                epoch,
            )
        writer.add_scalar('Train/n_views', current_views, epoch)
        writer.add_scalar('Train/Stage', current_stage, epoch)

        validation_metrics = None
        # 验证始终使用 final_view；只由验证 PSNR 选择 best checkpoint。
        if epoch % args.eval_every == 0 or epoch == total_epochs:
            validation_metrics = evaluate(
                model, val_loader, device, criterion,
                n_views=args.final_view,
                base_views=max(resolved_views),
            )
            selection_metric = float(validation_metrics['psnr'])
            print(
                f'  Val | E{epoch} PSNR={validation_metrics["psnr"]:.2f}dB '
                f'SSIM={validation_metrics["ssim"]:.4f}'
            )
            for name, value in validation_metrics.items():
                writer.add_scalar(f'Val/{name}', value, epoch)
            writer.add_scalar('Val/n_views', args.final_view, epoch)

            if selection_metric > best_metric:
                # 只有严格提升时覆盖 best，避免相同指标反复写盘。
                best_metric = selection_metric
                best_epoch = epoch
                payload = checkpoint_payload(
                    epoch=epoch, model=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler, args=args,
                    run_name=run_name, stage=current_stage,
                    current_train_views=current_views,
                    resolved_views=resolved_views,
                    best_metric=best_metric, best_epoch=best_epoch,
                    eval_metrics=validation_metrics,
                )
                torch.save(
                    payload, os.path.join(log_dir, best_checkpoint)
                )
                print(
                    f'  >> Best validation checkpoint '
                    f'(selection PSNR={best_metric:.2f}dB)'
                )

        # 周期 checkpoint 用于长训练中的故障恢复和回溯。
        if epoch % args.save_every == 0:
            periodic_checkpoint = checkpoint_name(
                'epoch', args.final_view, args.run_version, epoch
            )
            torch.save(
                checkpoint_payload(
                    epoch=epoch, model=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler, args=args,
                    run_name=run_name, stage=current_stage,
                    current_train_views=current_views,
                    resolved_views=resolved_views,
                    best_metric=best_metric, best_epoch=best_epoch,
                    eval_metrics=validation_metrics,
                ),
                os.path.join(log_dir, periodic_checkpoint),
            )
        last_epoch = epoch
        last_stage = current_stage
        last_views = current_views

    # 正常结束后保存最终 last 状态；它与验证最优 best 的用途不同。
    if optimizer is not None:
        torch.save(
            checkpoint_payload(
                epoch=last_epoch, model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, args=args,
                run_name=run_name, stage=last_stage,
                current_train_views=last_views,
                resolved_views=resolved_views,
                best_metric=best_metric, best_epoch=best_epoch,
            ),
            os.path.join(log_dir, last_checkpoint),
        )

    best_path = os.path.join(log_dir, best_checkpoint)
    if not os.path.isfile(best_path):
        if (
            args.resume
            and int(resume_checkpoint.get('epoch', -1)) == best_epoch
        ):
            best_path = args.resume
        else:
            raise FileNotFoundError(
                'Best validation checkpoint is missing; cannot perform the '
                'held-out test evaluation.'
            )
    # 测试集只在训练完成后，用验证集选出的 best 权重评估一次。
    best_payload = load_training_checkpoint(best_path, device)
    model.load_state_dict(best_payload['model_state'])
    best_loss_weights = get_stage_config(
        int(best_payload.get('stage', 3))
    )[0]
    criterion.set_weights(**best_loss_weights)
    test_metrics = evaluate(
        model, test_loader, device, criterion, n_views=args.final_view,
        base_views=max(resolved_views),
    )
    for name, value in test_metrics.items():
        writer.add_scalar(f'Test/{name}', value, total_epochs)
    writer.add_scalar('Test/n_views', args.final_view, total_epochs)
    with open(os.path.join(log_dir, 'test_metrics.json'), 'w') as handle:
        json.dump({
            'checkpoint': os.path.abspath(best_path),
            'best_val_epoch': best_epoch,
            'final_view': args.final_view,
            'metrics': test_metrics,
        }, handle, indent=2)
    writer.close()
    print(
        f'\nDone. Best validation PSNR: {best_metric:.2f}dB at epoch '
        f'{best_epoch}. Held-out test evaluated once at {args.final_view} views.'
    )


# =========================================================================
# 参数
# =========================================================================

if __name__=='__main__':
    p=argparse.ArgumentParser(description='SparseViewReconstruction training')
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--vol_size', type=int, nargs=3, default=(128,128,128))
    p.add_argument('--organ', type=str, default='thorax_fast')
    p.add_argument('--run_version', type=str, required=True,
                   help='Unique run version such as v3; existing non-empty runs are rejected')
    p.add_argument('--final_view', type=int, choices=(6,8,10), default=6,
                   help='Final deployment/evaluation view count')
    p.add_argument('--view_schedule', type=str, default=None,
                   help='Optional full high-to-low override, e.g. "60,48,24,12,6"')
    p.add_argument(
        '--source_views', type=int, default=-1,
        help='-1 loads every case in full and adapts per case; a positive '
             'value optionally asserts an identical source count',
    )
    p.add_argument('--stage1_epochs', type=int, default=200,
                   help='Stage 1 epochs at the curriculum maximum with EMA updates enabled')
    p.add_argument('--stage2_epochs_per_view', type=int, default=40, help='阶段2 每个视角的训练轮数')
    p.add_argument('--stage3_epochs', type=int, default=100,
                   help='Stage 3 epochs at final_view with quantizer frozen')
    p.add_argument('--proj_size', type=int, nargs=2, default=(128,128))
    p.add_argument('--n_decoder_ups', type=int, default=1)
    p.add_argument(
        '--transformer_layers', type=int, default=4,
        help='Number of cross-view TransformerEncoderLayer blocks',
    )
    p.add_argument('--hf_codebook_size', type=int, default=512)
    p.add_argument('--mf_codebook_size', type=int, default=256)
    p.add_argument(
        '--kmeans_iters', type=int, default=10,
        help='Lloyd iterations for first Stage-1 codebook initialization',
    )
    p.add_argument(
        '--kmeans_samples_per_code', type=int, default=4,
        help='Uniform current-feature samples per codeword for K-means',
    )
    p.add_argument(
        '--kmeans_init_batches', type=int, default=8,
        help='Collect K-means samples across this many shuffled train batches',
    )
    p.add_argument(
        '--dead_code_threshold', type=float, default=0.1,
        help='EMA cluster-size threshold below which a code is reinitialized',
    )
    p.add_argument(
        '--dead_code_check_interval', type=int, default=100,
        help='Check dead codes every N EMA-updating forward passes',
    )
    p.add_argument(
        '--dead_code_warmup_steps', type=int, default=100,
        help='Wait this many EMA-updating forward passes before dead-code checks',
    )
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--grad_accum', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--no_amp', action='store_false', dest='amp')
    p.add_argument('--log_dir', type=str, default='./logs')
    p.add_argument('--eval_every', type=int, default=10)
    p.add_argument('--save_every', type=int, default=50)
    p.add_argument('--ct_range', type=float, nargs=2, default=(-1000.0,1000.0),
                   help='HU range represented by normalized CT labels [0,1]')
    p.add_argument('--resume', type=str, default=None,
                   help='Resume from a full best/periodic/last checkpoint')
    args=p.parse_args()
    train(args)
