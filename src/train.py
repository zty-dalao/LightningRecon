"""
训练脚本 v3: 三阶段训练 (预训练码本 → 冻结码本微调 → 推理)

阶段1 (前 stage1_epochs): 全视角, 训练全部权重
阶段2 (剩余 epoch):      冻结码本, 渐进Mask微调

用法: python src/train.py --data_root data/thorax_fast --epochs 400 --stage1_epochs 100
"""

import os, sys, argparse, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import SparseViewReconstruction
from src.dataset import ThoraxCTDataset
from src.losses import ReconstructionLoss


def add_angle_encoding(projs, V_total, device):
    B, V, _, H, W = projs.shape
    theta = torch.linspace(0, 2 * torch.pi, V_total + 1, device=device)[:V]
    s = torch.sin(theta).view(1, -1, 1, 1, 1).expand(B, -1, 1, H, W)
    c = torch.cos(theta).view(1, -1, 1, 1, 1).expand(B, -1, 1, H, W)
    return torch.cat([projs, s, c], dim=2)


def subsample_projections(projs, n_keep, device):
    if projs.dim() != 5:
        raise ValueError(f'Expected 5D projections tensor, got shape {tuple(projs.shape)}')

    B, V_total, C, H, W = projs.shape
    if n_keep >= V_total:
        return projs[:, :V_total].contiguous()

    idx = torch.randperm(V_total, device=device)[:n_keep].sort().values
    return projs.index_select(1, idx).contiguous()


def _psnr(p, t, mask=None):
    """PSNR: 若提供 mask, 仅计算 mask>0 区域"""
    if mask is not None:
        mask = mask.bool()
        if mask.sum() == 0:
            return float('nan')
        mse = ((p[mask] - t[mask]) ** 2).mean()
    else:
        mse = torch.mean((p - t) ** 2)
    return float('inf') if mse == 0 else 20 * torch.log10(1.0 / torch.sqrt(mse))


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval(); m = {'psnr':0,'psnr_mask':0,'ssim':0,'lap':0,'struct':0,'total':0,'c':0,'c_mask':0}
    for b in loader:
        projs = b['projs'].to(device); ct = b['ct'].to(device)
        mask = b.get('mask')
        if mask is not None: mask = mask.to(device)
        projs_enc = add_angle_encoding(projs, projs.shape[1], device)
        pred, vq, _ = model(projs_enc)
        ct_a = nn.functional.interpolate(ct, size=pred.shape[2:], mode='trilinear')
        loss = criterion(pred, ct_a, vq); B = projs.shape[0]
        m['psnr'] += _psnr(pred, ct_a).item() * B
        if mask is not None:
            mask_a = nn.functional.interpolate(mask, size=pred.shape[2:], mode='nearest')
            psnr_m = _psnr(pred, ct_a, mask_a)
            if not np.isnan(float(psnr_m)):
                m['psnr_mask'] += psnr_m * B
                m['c_mask'] += B
        for k in ['ssim','lap','struct','total']: m[k]+=loss[k].item()*B
        m['c']+=B
    result = {k: v / m['c'] for k, v in m.items() if k not in ('c', 'c_mask')}
    if m['c_mask'] > 0:
        result['psnr_mask'] = m['psnr_mask'] / m['c_mask']
    return result


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据集加载: n_views 设为阶段一和阶段二视角上限的较大值
    # 数据集内部随机采样，每次 __getitem__ 得到不同视角组合
    load_views = max(args.stage1_views, args.max_views)
    proj_size = tuple(args.proj_size)
    ts = ThoraxCTDataset(data_root=args.data_root, split='train', n_views=load_views, proj_size=proj_size, vol_size=args.vol_size)
    vs = ThoraxCTDataset(data_root=args.data_root, split='test', n_views=load_views, proj_size=proj_size, vol_size=args.vol_size)
    tl = DataLoader(ts, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    vl = DataLoader(vs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f'Train:{len(ts)} Test:{len(vs)}')

    model = SparseViewReconstruction(n_decoder_ups=args.n_decoder_ups).to(device)
    print(f'Model: {sum(p.numel() for p in model.parameters()):,} params')

    criterion = ReconstructionLoss(w_lap=args.w_lap, w_struct=args.w_struct, w_vq=args.w_vq)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = GradScaler() if args.amp else None

    organ = getattr(args, 'organ', 'thorax')
    out_res = 128 * (2 ** args.n_decoder_ups)                        # 128→256 or 128→512
    log_dir = os.path.join(args.log_dir, f'{organ}_{args.train_views}view_{out_res}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))
    json.dump(vars(args)|{'train_cases':len(ts)}, open(os.path.join(log_dir,'config.json'),'w'), indent=2)

    best_psnr, V_total = -float('inf'), max(ts.max_views, vs.max_views)
    print(f'Max views across dataset: {V_total}')

    for epoch in range(1, args.epochs+1):
        # ---- 阶段切换 ----
        if epoch == 1: print('[阶段一] 第1轮开始，预训练码本...')
        if epoch == args.stage1_epochs + 1:
            model.freeze_codebooks()
            print(f'[阶段二] 第{epoch}轮开始，码本已冻结，微调中...')

        model.train()
        ep = {'total':0,'lap':0,'struct':0,'vq':0,'ssim':0}

        # 渐进Mask (阶段2才启动)
        if epoch <= args.stage1_epochs:
            n_keep = min(V_total, args.stage1_views)  # 阶段一: 每batch随机采样, 多epoch覆盖全视角
        else:
            progress = min(1.0, (epoch-args.stage1_epochs)/(args.epochs*args.trans_ratio))
            keep_ratio = 1.0 - (1.0-args.target_keep)*progress
            n_keep = max(args.train_views, min(int(V_total*keep_ratio), args.max_views))

        accum = max(1, args.grad_accum) if epoch <= args.stage1_epochs else 1
        opt.zero_grad()
        for bi, batch in enumerate(tl):
            projs = batch['projs'].to(device); ct = batch['ct'].to(device)
            if projs.dim() != 5:
                projs = projs.unsqueeze(1) if projs.dim() == 4 else projs
            actual_views = projs.shape[1]
            if n_keep < actual_views:
                projs = subsample_projections(projs, n_keep, device)
            else:
                projs = projs[:, :actual_views]
            projs_enc = add_angle_encoding(projs, projs.shape[1], device)

            if args.amp:
                with autocast():
                    pred, vq, _ = model(projs_enc)
                    ct_a = nn.functional.interpolate(ct, size=pred.shape[2:], mode='trilinear')
                    loss = criterion(pred, ct_a, vq)
                scaler.scale(loss['total'] / accum).backward()
            else:
                pred, vq, _ = model(projs_enc)
                ct_a = nn.functional.interpolate(ct, size=pred.shape[2:], mode='trilinear')
                loss = criterion(pred, ct_a, vq)
                (loss['total'] / accum).backward()

            if (bi + 1) % accum == 0 or bi == len(tl) - 1:
                if args.amp:
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    scaler.step(opt); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    opt.step()
                opt.zero_grad()

            for k in ep: ep[k] += loss[k].item()
        sch.step()

        nb = len(tl); avg = {k:v/nb for k,v in ep.items()}; lr = opt.param_groups[0]['lr']
        stage = 1 if epoch<=args.stage1_epochs else 2
        msg = f'第{epoch:4d}轮 E{epoch:4d} S{stage} lr={lr:.2e} V={n_keep} | 总损失={avg["total"]:.4f} 拉普拉斯={avg["lap"]:.4f} 结构={avg["struct"]:.4f} 码本={avg["vq"]:.4f}'
        print(msg)
        tag_map = {'total': '总损失 Total Loss', 'lap': '拉普拉斯损失 Laplacian',
                   'struct': '结构损失 Structure', 'vq': '码本损失 VQ', 'ssim': 'SSIM'}
        for k in avg: writer.add_scalar(f'Train/{tag_map.get(k, k)}', avg[k], epoch)
        writer.add_scalar('Train/学习率 LR', lr, epoch)
        writer.add_scalar('Train/视角数 n_views', n_keep, epoch)

        if epoch%args.eval_every==0 or epoch==args.epochs:
            em = evaluate(model, vl, device, criterion)
            psnr_mask_str = f' PSNR_mask={em.get("psnr_mask", float("nan")):.2f}dB' if 'psnr_mask' in em else ''
            print(f'  验证 | 第{epoch}轮 PSNR={em["psnr"]:.2f}dB{psnr_mask_str} SSIM={em["ssim"]:.4f} 拉普拉斯={em["lap"]:.4f} 结构={em["struct"]:.4f}')
            eval_tag = {'psnr': 'PSNR', 'psnr_mask': 'PSNR (Mask)', 'ssim': 'SSIM', 'lap': '拉普拉斯损失 Laplacian',
                        'struct': '结构损失 Structure', 'total': '总损失 Total Loss'}
            for k in em: writer.add_scalar(f'Eval/{eval_tag.get(k, k)}', em[k], epoch)
            if em['psnr']>best_psnr:
                best_psnr=em['psnr']
                torch.save({'epoch':epoch,'model_state':model.state_dict(),'best_psnr':best_psnr}, os.path.join(log_dir,'best_model.pth'))

        if epoch%args.save_every==0:
            torch.save({'epoch':epoch,'model_state':model.state_dict(),'best_psnr':best_psnr}, os.path.join(log_dir,f'ckpt_{epoch:04d}.pth'))

    writer.close(); print(f'Done. Best PSNR:{best_psnr:.2f}dB | {log_dir}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--vol_size', type=int, nargs=3, default=(128,128,128))
    p.add_argument('--organ', type=str, default='thorax_fast')
    p.add_argument('--train_views', type=int, default=6)
    p.add_argument('--max_views', type=int, default=48)
    p.add_argument('--target_keep', type=float, default=0.012)
    p.add_argument('--trans_ratio', type=float, default=0.5, help='Mask增长率(阶段2内)')
    p.add_argument('--stage1_epochs', type=int, default=100, help='阶段1(预训练码本)的epoch数')
    p.add_argument('--stage1_views', type=int, default=64, help='阶段1每batch最多视角数(随机采样, 显存限制)')
    p.add_argument('--proj_size', type=int, nargs=2, default=(256,256), help='投影resize (H,W), 阶段一可用较小值如128,128')
    p.add_argument('--n_decoder_ups', type=int, default=1, help='解码器上采样次数(1=256³,2=512³)')
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--grad_accum', type=int, default=1, help='梯度累积步数(阶段一增大可等效更多视角)')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--no_amp', action='store_false', dest='amp')
    p.add_argument('--w_lap', type=float, default=1.0)
    p.add_argument('--w_struct', type=float, default=0.3)
    p.add_argument('--w_vq', type=float, default=0.1)
    p.add_argument('--log_dir', type=str, default='./logs')
    p.add_argument('--eval_every', type=int, default=10)
    p.add_argument('--save_every', type=int, default=50)
    args = p.parse_args()
    train(args)
