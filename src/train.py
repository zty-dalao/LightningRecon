"""
训练脚本: 稀疏视角 CT 重建模型 (SparseViewReconstruction)。

渐进式训练策略 (来自 mymodel.md):
  阶段1 (epoch  1-50):   只用 CBCT 监督 — 学习解剖结构
  阶段2 (epoch 51-100):  逐渐引入 pCT 监督 — 校正 CT 值
  阶段3 (epoch 101-300): 完整损失 — 联合微调
  阶段4 (epoch 301+):    冻结低频基座 — 微调其余模块

数据来源: data/paired/
  projection_npy/{case_id}/Proj_*.npy  — 491 张投影
  cbct/{case_id}.nii.gz                 — CBCT 重建体
  ct/{case_id}.nii.gz                   — pCT 标签体

用法示例:
  python src/train.py \
      --data_root /home/zty20020112/workspace/LightningRecon/data/paired \
      --epochs 400 \
      --batch_size 1 \
      --lr 1e-4 \
      --n_views 491 \
      --n_cond_views 6
"""

import os                                                           # 文件路径操作
import sys                                                          # 系统路径管理
import argparse                                                     # 命令行参数解析
import json                                                         # JSON 配置读写

import torch                                                        # PyTorch 核心
import torch.nn as nn                                               # 神经网络模块
from torch.utils.data import DataLoader                             # 数据加载器
from torch.utils.tensorboard import SummaryWriter                   # TensorBoard 日志
from torch.cuda.amp import GradScaler, autocast                     # 混合精度训练 (AMP)

# 将项目根目录加入 sys.path，确保 src 包可被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SparseViewReconstruction                  # 稀疏视角重建模型
from src.dataset import PairedCBCTDataset                        # 配对 CBCT-pCT 数据集
from src.losses import ReconstructionLoss                        # 双监督重建损失


# =========================================================================
# 评估指标
# =========================================================================

def _psnr(pred, target, data_range=1.0):
    """
    计算峰值信噪比 (Peak Signal-to-Noise Ratio)。

    PSNR = 20 * log10(MAX / sqrt(MSE))
    值越高表示重建质量越好，单位 dB。
    """
    mse = torch.mean((pred - target) ** 2)                          # 均方误差
    if mse == 0:                                                    # 完全一致
        return float('inf')                                         # 无穷大
    return 20 * torch.log10(data_range / torch.sqrt(mse))           # PSNR 公式


def _mae(pred, target):
    """计算平均绝对误差 (Mean Absolute Error)。"""
    return torch.mean(torch.abs(pred - target))                     # |pred - target| 的均值


# =========================================================================
# 评估函数: 在验证/测试集上计算指标
# =========================================================================

def evaluate(model, dataloader, device, criterion):
    """
    在给定 dataloader 上评估模型性能。

    Args:
        model:      模型实例
        dataloader: 验证/测试数据加载器
        device:     'cuda' 或 'cpu'
        criterion:  ReconstructionLoss 实例

    Returns:
        dict: {'psnr', 'ssim', 'mae', 'total_loss'} 平均值
    """
    model.eval()                                                    # 切换到评估模式 (关闭 BN/Dropout)
    metrics = {                                                     # 初始化累积器
        'psnr': 0.0, 'ssim': 0.0, 'mae': 0.0,
        'total_loss': 0.0, 'count': 0,
    }

    with torch.no_grad():                                           # 禁用梯度计算 (省显存)
        for batch in dataloader:                                    # 遍历每个 batch
            projs = batch['projs'].to(device)                       # 投影图 → GPU/CPU
            ct   = batch['ct'].to(device)                           # pCT 标签 → GPU/CPU
            cbct = batch['cbct'].to(device)                         # CBCT 标签 → GPU/CPU

            volume_pred, low_freq, vq_loss, perplexity = model(projs)  # 模型前向

            # 对齐尺寸: 模型输出可能与标签尺寸不同，用三线性插值对齐
            if volume_pred.shape != ct.shape:                       # 尺寸不匹配
                volume_pred = nn.functional.interpolate(            # 三线性插值
                    volume_pred, size=ct.shape[2:],
                    mode='trilinear', align_corners=False
                )

            loss_dict = criterion(volume_pred, ct, cbct,            # 计算损失
                                  low_freq, vq_loss, perplexity)

            B = projs.shape[0]                                      # 当前 batch 大小
            metrics['psnr'] += _psnr(volume_pred, ct).item() * B    # 累加 PSNR
            metrics['ssim'] += loss_dict['ssim'].item() * B          # 累加 SSIM
            metrics['mae'] += _mae(volume_pred, ct).item() * B      # 累加 MAE
            metrics['total_loss'] += loss_dict['total'].item() * B  # 累加总损失
            metrics['count'] += B                                   # 累加样本数

    # 取平均
    for k in ['psnr', 'ssim', 'mae', 'total_loss']:                 # 遍历每个指标
        metrics[k] /= metrics['count']                              # 除以样本总数
    return metrics                                                  # 返回平均指标


# =========================================================================
# 训练主函数
# =========================================================================

def train(args):
    """执行完整训练流程。"""
    # ---- 设备选择 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 优先 GPU
    print(f'Device: {device}')                                      # 打印设备信息

    # ---- 数据集 ----
    train_set = PairedCBCTDataset(                                  # 训练集
        data_root=args.data_root,                                   # paired/ 路径
        split='train',                                              # 训练划分
        n_views=args.n_views,                                       # 投影数 (491)
        proj_size=(256, 256),                                       # 投影 resize 尺寸
        vol_size=(128, 128, 128),                                   # 体素 resize 尺寸
    )
    test_set = PairedCBCTDataset(                                   # 测试/验证集
        data_root=args.data_root,
        split='test',
        n_views=args.n_views,                                       # 评估也用全量投影
        proj_size=(256, 256),
        vol_size=(128, 128, 128),
    )

    train_loader = DataLoader(                                      # 训练数据加载器
        train_set,                                                   # 训练数据集
        batch_size=args.batch_size,                                  # batch 大小
        shuffle=True,                                                # 随机打乱
        num_workers=args.num_workers,                                # 并行加载线程数
        pin_memory=True,                                             # 锁页内存 (加速 GPU 传输)
    )
    test_loader = DataLoader(                                       # 测试数据加载器
        test_set,                                                    # 测试数据集
        batch_size=args.batch_size,                                  # batch 大小
        shuffle=False,                                               # 不打乱 (保证可复现)
        num_workers=args.num_workers,                                # 并行加载线程数
        pin_memory=True,                                             # 锁页内存
    )

    print(f'Train: {len(train_set)} cases, Test: {len(test_set)} cases')  # 打印数据量

    # ---- 模型 ----
    model = SparseViewReconstruction(                               # 实例化模型
        max_views=args.n_views,                                      # 最大视角数 (训练时在此范围内随机采样)
        vol_ch=args.vol_ch,                                          # 3D 特征体通道数
        base_ch=args.base_ch,                                        # 低频基座通道数
        codebook_size=args.codebook_size,                            # 码本大小
        codebook_dim=args.codebook_dim,                              # 码本向量维度
    ).to(device)                                                     # 移至 GPU/CPU

    total_params = sum(p.numel() for p in model.parameters())       # 总参数量
    trainable_params = sum(                                         # 可训练参数量
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f'Model params: {total_params:,} total, '                 # 打印参数统计
          f'{trainable_params:,} trainable')

    # ---- 损失函数 ----
    criterion = ReconstructionLoss(                                 # 双监督重建损失
        w_cbct=1.0,                                                  # CBCT 损失权重
        w_pct=1.0,                                                   # pCT 损失权重
        w_smooth=0.01,                                               # 平滑正则权重
        w_vq=0.1,                                                    # VQ 承诺损失权重
        w_sparse=0.001,                                              # 稀疏正则权重
        w_grad=0.05,                                                 # 梯度损失权重
        w_hist=0.1,                                                  # 直方图损失权重
    )

    # ---- 优化器 & 学习率调度器 ----
    optimizer = torch.optim.AdamW(                                  # AdamW 优化器
        model.parameters(),                                          # 所有可训练参数
        lr=args.lr,                                                  # 初始学习率
        weight_decay=1e-5,                                           # 权重衰减 (L2 正则)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(         # 余弦退火调度器
        optimizer, T_max=args.epochs                                 # 周期 = 总 epoch 数
    )

    scaler = GradScaler() if args.amp else None                     # AMP 梯度缩放器 (可选)

    # ---- 日志目录: logs/{organ}_{train_views}view/ ----
    # 例如: logs/thorax_6view/
    # 同一器官+视角的训练 loss 全部汇总于此，方便 TensorBoard 查看
    organ = getattr(args, 'organ', 'thorax')                        # 器官名 (默认 thorax)
    nv_str = str(args.train_views) if args.train_views > 0 else 'full'  # 视角数 (0=全视角)
    log_name = f'{organ}_{nv_str}view'                              # 日志子目录名
    log_dir = os.path.join(args.log_dir, log_name)                  # 日志路径 (无时间戳)
    os.makedirs(log_dir, exist_ok=True)                             # 创建目录 (递归)

    # ---- TensorBoard 日志器 ----
    tb_dir = os.path.join(log_dir, 'tensorboard')                   # TensorBoard 子目录
    writer = SummaryWriter(log_dir=tb_dir)                          # 创建 SummaryWriter

    # 保存运行配置到 JSON
    config_dict = vars(args).copy()                                 # 命令行参数拷贝
    config_dict['train_cases'] = len(train_set)                     # 训练集大小
    config_dict['test_cases'] = len(test_set)                       # 测试集大小
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:      # 写入配置
        json.dump(config_dict, f, indent=2)

    best_psnr = -float('inf')                                       # 最佳 PSNR (初始化为负无穷)
    log_lines = []                                                  # 训练日志文本行

    # ---- 三阶段课程学习参数 ----
    import random                                                   # 随机数 (批次级决策)
    warmup_epochs = int(args.epochs * args.warmup_ratio)            # 预热期结束 epoch (默认前 20%)
    trans_epochs = int(args.epochs * args.trans_ratio)              # 过渡期结束 epoch (默认前 70%)
    total_sparse = 0                                                # 稀疏视角 batch 计数
    total_full = 0                                                  # 全视角 batch 计数

    # =====================================================================
    # 训练循环
    # =====================================================================
    for epoch in range(1, args.epochs + 1):                         # 从第 1 到第 N 个 epoch
        model.train()                                               # 切换到训练模式
        epoch_losses = {k: 0.0 for k in                             # 初始化 epoch 损失累积器
                        ['total', 'cbct', 'pct', 'ssim', 'vq', 'smooth']}
        epoch_sparse = 0                                            # 本 epoch 稀疏视角 batch 数
        epoch_full = 0                                              # 本 epoch 全视角 batch 数

        # ---- 阶段1: 只用 CBCT → 阶段2: 引入 pCT → 阶段3: 完整 → 阶段4: 冻结基座 ----
        if epoch <= 50:                                             # 阶段1: 只用 CBCT
            criterion.w_pct = 0.0                                   # pCT 损失权重 = 0
            criterion.w_cbct = 1.0                                  # CBCT 损失权重 = 1
        elif epoch <= 100:                                          # 阶段2: 逐渐引入 pCT
            criterion.w_pct = 0.5                                   # pCT 半权重
            criterion.w_cbct = 1.0                                  # CBCT 全权重
        elif epoch <= 300:                                          # 阶段3: 完整联合损失
            criterion.w_pct = 1.0                                   # pCT 全权重
            criterion.w_cbct = 1.0                                  # CBCT 全权重
        else:                                                       # 阶段4: 冻结低频基座
            criterion.w_pct = 1.0                                   # pCT 全权重
            criterion.w_cbct = 1.0                                  # CBCT 全权重
            if epoch == 301:                                        # 仅在进入阶段4时冻结一次
                model.low_freq_base.base_feat.requires_grad = False # 冻结通用解剖基座
                print('  [Stage 4] Frozen low_freq_base.base_feat')

        # ---- 课程学习: 计算本 epoch 稀疏视角概率 P(6view) ----
        if epoch <= warmup_epochs:                                  # 阶段A: 预热期 (纯全视角)
            sparse_prob = 0.0                                       # P(6view) = 0
        elif epoch <= trans_epochs:                                 # 阶段B: 过渡期 (线性增长)
            progress = (epoch - warmup_epochs) / (trans_epochs - warmup_epochs)  # [0, 1]
            sparse_prob = args.max_sparse_prob * progress           # 0 → max_sparse_prob
        else:                                                       # 阶段C: 微调期 (固定概率)
            sparse_prob = args.max_sparse_prob                      # P(6view) = max_sparse_prob

        # ---- 遍历训练 batch ----
        for batch_idx, batch in enumerate(train_loader):            # 遍历每个 batch
            projs = batch['projs'].to(device)                       # 投影 (全部 491) → device
            ct   = batch['ct'].to(device)                           # pCT → device
            cbct = batch['cbct'].to(device)                         # CBCT → device

            # ---- 批次级决策: 整个 batch 统一用全视角或稀疏视角 ----
            if args.train_views <= 0 or random.random() >= sparse_prob:
                # 全视角模式
                n_sample = None                                     # None = 使用全部 491
                loss_w = args.full_loss_weight                      # 全视角损失权重 (较大，稳定梯度)
                epoch_full += 1
            else:
                # 稀疏视角模式
                n_sample = args.train_views                         # 随机选 train_views 个 (如 6)
                loss_w = 1.0                                        # 稀疏视角损失权重 (基础)
                epoch_sparse += 1

            optimizer.zero_grad()                                   # 清零梯度

            if args.amp:                                            # 混合精度训练路径
                with autocast():                                    # 自动混合精度上下文
                    volume_pred, low_freq, vq_loss, perp = model(   # 前向
                        projs, n_sample_views=n_sample
                    )
                    if volume_pred.shape != ct.shape:               # 尺寸对齐
                        volume_pred = nn.functional.interpolate(
                            volume_pred, size=ct.shape[2:],
                            mode='trilinear', align_corners=False
                        )
                    loss_dict = criterion(                          # 计算损失
                        volume_pred, ct, cbct, low_freq, vq_loss, perp
                    )
                    loss_dict['total'] = loss_dict['total'] * loss_w  # 损失加权
                scaler.scale(loss_dict['total']).backward()         # 缩放损失 + 反向传播
                scaler.unscale_(optimizer)                          # 反缩放梯度
                torch.nn.utils.clip_grad_norm_(                     # 梯度裁剪 (防梯度爆炸)
                    model.parameters(), 1.0
                )
                scaler.step(optimizer)                              # 优化器步进
                scaler.update()                                     # 更新缩放因子
            else:                                                   # 标准精度训练路径
                volume_pred, low_freq, vq_loss, perp = model(       # 前向
                    projs, n_sample_views=n_sample
                )
                if volume_pred.shape != ct.shape:                   # 尺寸对齐
                    volume_pred = nn.functional.interpolate(
                        volume_pred, size=ct.shape[2:],
                        mode='trilinear', align_corners=False
                    )
                loss_dict = criterion(                              # 计算损失
                    volume_pred, ct, cbct, low_freq, vq_loss, perp
                )
                (loss_dict['total'] * loss_w).backward()            # 加权损失反向传播
                torch.nn.utils.clip_grad_norm_(                     # 梯度裁剪
                    model.parameters(), 1.0
                )
                optimizer.step()                                    # 优化器步进

            # 累积 epoch 统计
            for k in epoch_losses:                                  # 遍历每个损失键
                epoch_losses[k] += loss_dict[k].item()              # 累加标量值

        scheduler.step()                                            # 学习率调度器步进
        total_sparse += epoch_sparse                                # 累计稀疏 batch
        total_full += epoch_full                                    # 累计全视角 batch

        # ---- 每 epoch 输出 ----
        n_batches = len(train_loader)                               # batch 数量
        avg_losses = {k: v / n_batches                              # 计算平均损失
                      for k, v in epoch_losses.items()}
        lr_now = optimizer.param_groups[0]['lr']                    # 当前学习率

        train_msg = (                                               # 格式化训练状态
            f'Epoch {epoch:4d}/{args.epochs} | '
            f'lr={lr_now:.2e} | '
            f'P6v={sparse_prob:.2f} | '
            f'Loss={avg_losses["total"]:.4f} | '
            f'CBCT={avg_losses["cbct"]:.4f} | '
            f'pCT={avg_losses["pct"]:.4f} | '
            f'SSIM={avg_losses["ssim"]:.4f} | '
            f'S/F={epoch_sparse}/{epoch_full}'                      # 稀疏/全视角 batch 数
        )
        print(train_msg)                                            # 打印到终端
        log_lines.append(train_msg)                                 # 加入日志

        # ---- TensorBoard 记录 (训练损失) ----
        writer.add_scalar('Train/Loss_total', avg_losses['total'], epoch)  # 总损失
        writer.add_scalar('Train/Loss_cbct',  avg_losses['cbct'],  epoch)  # CBCT 损失
        writer.add_scalar('Train/Loss_pct',   avg_losses['pct'],   epoch)  # pCT 损失
        writer.add_scalar('Train/SSIM',       avg_losses['ssim'],  epoch)  # SSIM
        writer.add_scalar('Train/VQ_loss',    avg_losses['vq'],    epoch)  # VQ 损失
        writer.add_scalar('Train/LR',         lr_now,              epoch)  # 学习率
        writer.add_scalar('Train/P_sparse',   sparse_prob,         epoch)  # 稀疏视角概率
        writer.add_scalar('Train/Sparse_batches', epoch_sparse,    epoch)  # 稀疏 batch 数
        writer.add_scalar('Train/Full_batches',   epoch_full,      epoch)  # 全视角 batch 数

        # ---- 定期评估 ----
        if epoch % args.eval_every == 0 or epoch == args.epochs:    # 每 eval_every 个 epoch
            eval_metrics = evaluate(model, test_loader, device, criterion)  # 评估
            eval_msg = (                                            # 格式化评估结果
                f'  Eval [{epoch}]: '
                f'PSNR={eval_metrics["psnr"]:.2f}dB | '
                f'SSIM={eval_metrics["ssim"]:.4f} | '
                f'MAE={eval_metrics["mae"]:.4f} | '
                f'Loss={eval_metrics["total_loss"]:.4f}'
            )
            print(eval_msg)                                         # 打印评估结果
            log_lines.append(eval_msg)                              # 加入日志

            # ---- TensorBoard 记录 (评估指标) ----
            writer.add_scalar('Eval/PSNR', eval_metrics['psnr'], epoch)       # PSNR
            writer.add_scalar('Eval/SSIM', eval_metrics['ssim'], epoch)       # SSIM
            writer.add_scalar('Eval/MAE',  eval_metrics['mae'],  epoch)       # MAE
            writer.add_scalar('Eval/Loss', eval_metrics['total_loss'], epoch) # 评估损失

            # 保存最佳模型
            if eval_metrics['psnr'] > best_psnr:                    # PSNR 更高 = 更好
                best_psnr = eval_metrics['psnr']                     # 更新最佳 PSNR
                ckpt = {                                            # 检查点字典
                    'epoch': epoch,                                  # 当前 epoch
                    'model_state': model.state_dict(),               # 模型权重
                    'optimizer_state': optimizer.state_dict(),       # 优化器状态
                    'best_psnr': best_psnr,                          # 最佳 PSNR
                }
                torch.save(ckpt,                                    # 保存检查点
                           os.path.join(log_dir, 'best_model.pth'))
                print(f'  → Saved best model (PSNR={best_psnr:.2f}dB)')

        # ---- 定期保存检查点 (用于断点续训) ----
        if epoch % args.save_every == 0:                            # 每 save_every 个 epoch
            ckpt = {                                                # 检查点字典
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_psnr': best_psnr,
            }
            torch.save(ckpt,                                        # 保存检查点
                       os.path.join(log_dir, f'checkpoint_ep{epoch:04d}.pth'))

    # ---- 训练结束: 保存日志 & 关闭 TensorBoard ----
    with open(os.path.join(log_dir, 'train.log'), 'w') as f:        # 打开日志文件
        f.write('\n'.join(log_lines))                               # 写入所有日志行
    writer.close()                                                  # 关闭 TensorBoard writer

    print(f'\nTraining finished. Best PSNR: {best_psnr:.2f}dB')    # 打印最终结果
    print(f'Logs saved to {log_dir}')                               # 打印日志路径
    print(f'TensorBoard: tensorboard --logdir {tb_dir}')            # 提示查看命令


# =========================================================================
# 命令行参数 & 入口
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(                               # 参数解析器
        description='Train SparseViewReconstruction model.'
    )

    # ---- 数据参数 ----
    parser.add_argument('--data_root', type=str, required=True,     # 数据根目录
                        help='paired/ 目录路径 (含 projection_npy/, cbct/, ct/)')
    parser.add_argument('--n_views', type=int, default=491,         # 每例投影数
                        help='每病例总投影数 (默认: 491)')
    parser.add_argument('--train_views', type=int, default=6,       # 稀疏训练视角数
                        help='稀疏训练时使用的视角数 (默认: 6, 设 0 表示只用全视角)')

    # ---- 课程学习参数 (三阶段渐进式) ----
    parser.add_argument('--warmup_ratio', type=float, default=0.2,  # 预热期占比
                        help='纯全视角预热期占总 epoch 比例 (默认: 0.2)')
    parser.add_argument('--trans_ratio', type=float, default=0.7,   # 过渡期结束占比
                        help='过渡期结束占总 epoch 比例 (默认: 0.7)')
    parser.add_argument('--max_sparse_prob', type=float, default=0.7,  # 最大稀疏概率
                        help='微调期稀疏视角最大概率 (默认: 0.7)')
    parser.add_argument('--full_loss_weight', type=float, default=10.0,  # 全视角损失权重
                        help='全视角时损失乘的权重，平衡 Loss 尺度 (默认: 10.0)')

    parser.add_argument('--organ', type=str, default='thorax',      # 器官名称
                        help='器官名称，用于日志目录命名 (默认: thorax)')

    # ---- 模型参数 ----
    parser.add_argument('--vol_ch', type=int, default=64,           # 3D 特征体通道
                        help='3D 特征体通道数 (默认: 64)')
    parser.add_argument('--base_ch', type=int, default=32,          # 基座通道数
                        help='低频基座通道数 (默认: 32)')
    parser.add_argument('--codebook_size', type=int, default=1024,  # 码本大小
                        help='VQ 码本条目数 (默认: 1024)')
    parser.add_argument('--codebook_dim', type=int, default=64,     # 码本维度
                        help='VQ 码本向量维度 (默认: 64)')

    # ---- 训练参数 ----
    parser.add_argument('--epochs', type=int, default=400,          # 训练轮数
                        help='训练总 epoch 数 (默认: 400)')
    parser.add_argument('--batch_size', type=int, default=1,        # batch 大小
                        help='batch 大小 (默认: 1, 3D 模型显存大)')
    parser.add_argument('--lr', type=float, default=1e-4,           # 学习率
                        help='初始学习率 (默认: 1e-4)')
    parser.add_argument('--num_workers', type=int, default=4,       # 数据加载线程
                        help='DataLoader 并行线程数 (默认: 4)')
    parser.add_argument('--amp', action='store_true', default=True, # 混合精度
                        help='使用自动混合精度训练 (默认: True)')
    parser.add_argument('--no_amp', action='store_false', dest='amp',  # 禁用 AMP
                        help='禁用混合精度')

    # ---- 日志参数 ----
    parser.add_argument('--log_dir', type=str,                      # 日志目录
                        default='./logs',
                        help='日志保存根目录 (默认: ./logs)')
    parser.add_argument('--eval_every', type=int, default=10,       # 评估频率
                        help='每 N 个 epoch 评估一次 (默认: 10)')
    parser.add_argument('--save_every', type=int, default=50,       # 保存频率
                        help='每 N 个 epoch 保存检查点 (默认: 50)')

    args = parser.parse_args()                                      # 解析命令行参数
    train(args)                                                     # 执行训练
