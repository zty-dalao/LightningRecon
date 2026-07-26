"""
训练脚本 v4: 三阶段训练 + 平滑视角衰减 + Charbonnier 主损失

阶段1 (Stage1): 64 views, 码本学习, L_img 主导
阶段2 (Stage2): 平滑降 view (64→56→48→...→6), 码本低LR
阶段3 (Stage3): 6 views 固定微调, 码本冻结

核心改动:
  - L_img (Charbonnier, mask 内) 为主损失 (w=1.0)
  - 视角平滑递减，避免 64→6 跳变
  - 按 Eval Mask PSNR 选最佳 checkpoint
  - Warmup + CosineAnnealingLR, 每阶段重置
  - 分组学习率 (encoder / codebook / decoder)

用法:
  python src/train.py --data_root /root/autodl-tmp/thorax \
      --stage1_epochs 150 --stage2_epochs_per_view 30 --stage3_epochs 100
"""

import os, sys, argparse, json, math
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import SparseViewReconstruction
from src.dataset import ThoraxCTDataset
from src.losses import ReconstructionLoss


# =========================================================================
# 工具
# =========================================================================

def add_angle_encoding(projs, V_total, device):
    B, V, _, H, W = projs.shape
    theta = torch.linspace(0, 2 * torch.pi, V_total + 1, device=device)[:V]
    s = torch.sin(theta).view(1, -1, 1, 1, 1).expand(B, -1, 1, H, W)
    c = torch.cos(theta).view(1, -1, 1, 1, 1).expand(B, -1, 1, H, W)
    return torch.cat([projs, s, c], dim=2)


def subsample_projections(projs, n_keep, device):
    if projs.dim() != 5:
        raise ValueError(f'Expected 5D, got {tuple(projs.shape)}')
    B, V_total, C, H, W = projs.shape
    if n_keep >= V_total:
        return projs[:, :V_total].contiguous()
    idx = torch.randperm(V_total, device=device)[:n_keep].sort().values
    return projs.index_select(1, idx).contiguous()


def _psnr(p, t, mask=None):
    if mask is not None:
        mask = mask.bool()
        if mask.sum() == 0:
            return float('nan')
        mse = ((p[mask] - t[mask]) ** 2).mean()
    else:
        mse = torch.mean((p - t) ** 2)
    return float('inf') if mse == 0 else 20 * torch.log10(1.0 / torch.sqrt(mse))


def _mae(p, t, mask):
    mask = mask.bool()
    if mask.sum() == 0:
        return float('nan')
    return (p[mask] - t[mask]).abs().mean().item()


# =========================================================================
# 视角调度
# =========================================================================

def auto_decay_views(start, end, steps):
    """
    根据步长序列自动生成视角衰减序列。

    steps=[8,4,2] 意为:
      - 阶段 A: 步长 8, 从 start 递减, 直到 next_step*3 (如 4*3=12)
      - 阶段 B: 步长 4, 继续递减, 直到 next_step*3 (如 2*3=6)
      - 阶段 C: 步长 2, 递减到 end

    示例: start=64, end=6, steps=[8,4,2]
      → [56, 48, 40, 32, 24,  20, 16, 12,  10, 8, 6]
    """
    views = []
    current = start
    for i, step in enumerate(steps):
        # 当前步长适用到 step*3 (如步长8→24, 步长4→12, 步长2→6)
        lower = max(end, step * 3)
        while current - step >= lower:
            current -= step
            views.append(current)
    while current > end:
        current = max(end, current - steps[-1])
        views.append(current)
    return views


def build_view_schedule(args):
    """返回 [(start_epoch, end_epoch, n_views, stage), ...]"""
    schedule = []
    s1_end = args.stage1_epochs
    schedule.append((1, s1_end, args.stage1_views, 1))

    if args.stage2_view_decay:
        parts = [int(v) for v in args.stage2_view_decay.split(',')]
        # 自动检测: 元素少且值小 → 步长模式; 否则 → 显式列表
        if len(parts) <= 3 and all(p <= 16 for p in parts):
            views = auto_decay_views(args.stage1_views, args.train_views, parts)
        else:
            views = parts
        ep_per_view = args.stage2_epochs_per_view
        cur = s1_end + 1
        for nv in views:
            schedule.append((cur, cur + ep_per_view - 1, nv, 2))
            cur += ep_per_view

    if args.stage3_epochs > 0:
        s2_end = schedule[-1][1] if len(schedule) > 1 else s1_end
        schedule.append((s2_end + 1, s2_end + args.stage3_epochs, args.train_views, 3))

    return schedule


def get_stage_config(stage):
    """(loss_weights_dict, lr_dict, freeze_codebook)"""
    if stage == 1:
        return ({'w_img': 1.0, 'w_lap': 0.05, 'w_struct': 0.10, 'w_vq': 0.05},
                {'encoder': 1e-4, 'codebook': 1e-4}, False)
    elif stage == 2:
        return ({'w_img': 1.0, 'w_lap': 0.04, 'w_struct': 0.08, 'w_vq': 0.02},
                {'encoder': 5e-5, 'codebook': 5e-6}, False)
    else:
        return ({'w_img': 1.0, 'w_lap': 0.02, 'w_struct': 0.05, 'w_vq': 0.0},
                {'encoder': 1e-5, 'decoder': 2e-5}, True)


# =========================================================================
# 验证
# =========================================================================

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    m = {'psnr':0,'psnr_mask':0,'ssim':0,'mae_mask':0,
         'total':0,'img':0,'lap':0,'struct':0,'c':0,'c_mask':0}
    for batch in loader:
        projs=batch['projs'].to(device); ct=batch['ct'].to(device)
        mask=batch.get('mask')
        if mask is not None: mask=mask.to(device)
        projs_enc=add_angle_encoding(projs, projs.shape[1], device)
        pred, vq, _ = model(projs_enc)
        ct_a=nn.functional.interpolate(ct, size=pred.shape[2:], mode='trilinear')
        mask_a=nn.functional.interpolate(mask, size=pred.shape[2:], mode='nearest') if mask is not None else None
        loss=criterion(pred, ct_a, vq, mask_a); B=projs.shape[0]
        m['psnr']+=_psnr(pred, ct_a)*B
        if mask_a is not None:
            pm=_psnr(pred, ct_a, mask_a)
            if not np.isnan(float(pm)):
                m['psnr_mask']+=float(pm)*B; m['mae_mask']+=_mae(pred, ct_a, mask_a)*B; m['c_mask']+=B
        for k in ['ssim','total','img','lap','struct']:
            v=loss[k]; m[k]+=(v.item() if isinstance(v, torch.Tensor) else v)*B
        m['c']+=B
    return {'psnr':m['psnr']/m['c'],
            'psnr_mask':m['psnr_mask']/m['c_mask'] if m['c_mask']>0 else float('nan'),
            'mae_mask':m['mae_mask']/m['c_mask'] if m['c_mask']>0 else float('nan'),
            'ssim':m['ssim']/m['c'], 'total':m['total']/m['c'],
            'img':m['img']/m['c'], 'lap':m['lap']/m['c'], 'struct':m['struct']/m['c']}


# =========================================================================
# 主训练
# =========================================================================

def train(args):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集
    load_views=max(args.stage1_views,
        max(int(v) for v in args.stage2_view_decay.split(',')) if args.stage2_view_decay else 6)
    proj_size=tuple(args.proj_size)
    ts=ThoraxCTDataset(data_root=args.data_root, split='train', n_views=load_views,
                       proj_size=proj_size, vol_size=args.vol_size)
    vs=ThoraxCTDataset(data_root=args.data_root, split='test', n_views=load_views,
                       proj_size=proj_size, vol_size=args.vol_size)
    tl=DataLoader(ts, batch_size=args.batch_size, shuffle=True,
                  num_workers=args.num_workers, pin_memory=True)
    vl=DataLoader(vs, batch_size=args.batch_size, shuffle=False,
                  num_workers=args.num_workers, pin_memory=True)
    print(f'Train:{len(ts)} Test:{len(vs)}')
    V_total=max(ts.max_views, vs.max_views)
    print(f'Max views: {V_total}')

    # 视角调度
    view_schedule=build_view_schedule(args)
    total_epochs=view_schedule[-1][1] if view_schedule else args.epochs
    print(f'View schedule ({len(view_schedule)} phases, {total_epochs} epochs):')
    for s,e,nv,st in view_schedule:
        print(f'  E{s:4d}-{e:4d} ({e-s+1:3d}ep): {nv:2d}v, Stage {st}')

    # 模型 + 损失
    model=SparseViewReconstruction(n_decoder_ups=args.n_decoder_ups).to(device)
    print(f'Model: {sum(p.numel() for p in model.parameters()):,} params')
    criterion=ReconstructionLoss()

    # 日志
    organ=getattr(args,'organ','thorax_fast')
    out_res=128*(2**args.n_decoder_ups)
    log_dir=os.path.join(args.log_dir, f'{organ}_{args.train_views}view_{out_res}')
    os.makedirs(log_dir, exist_ok=True)
    writer=SummaryWriter(os.path.join(log_dir,'tensorboard'))
    json.dump(vars(args)|{'train_cases':len(ts),'total_epochs':total_epochs,'V_total':V_total},
              open(os.path.join(log_dir,'config.json'),'w'), indent=2)

    scaler=GradScaler() if args.amp else None
    best_mask_psnr=-float('inf'); best_epoch=0; prev_stage=None
    accum=max(1, args.grad_accum)

    for epoch in range(1, total_epochs+1):
        # 查找当前阶段
        cur_stage=None; cur_views=args.stage1_views
        for s,e,nv,st in view_schedule:
            if s<=epoch<=e: cur_stage,cur_views=st,nv; break
        if cur_stage is None: break

        # 阶段切换
        if cur_stage!=prev_stage:
            lw,lr_cfg,freeze_cb=get_stage_config(cur_stage)
            criterion.set_weights(**lw)
            print(f'\n{"="*60}')
            print(f'[Stage {cur_stage}] Epoch {epoch}: views={cur_views}')
            print(f'  Loss: img={lw["w_img"]} lap={lw["w_lap"]} struct={lw["w_struct"]} vq={lw["w_vq"]}')

            if freeze_cb:
                model.freeze_codebooks(); print(f'  Codebook: frozen')
            else:
                for cb in [model.codebook_hf, model.codebook_mf]: cb.unfreeze()
                print(f'  Codebook: trainable')

            # 分组LR优化器
            dec_ids=set(id(p) for p in model.decoder.parameters())
            if not freeze_cb and abs(lr_cfg.get('codebook',1e-4)-lr_cfg.get('encoder',1e-4))>1e-10:
                cb_ids=set(id(p) for p in list(model.codebook_hf.parameters())+list(model.codebook_mf.parameters()))
                other=[p for p in model.parameters() if id(p) not in cb_ids and id(p) not in dec_ids]
                opt=torch.optim.AdamW([
                    {'params':other,'lr':lr_cfg['encoder']},
                    {'params':list(model.codebook_hf.parameters())+list(model.codebook_mf.parameters()),'lr':lr_cfg['codebook']},
                    {'params':model.decoder.parameters(),'lr':lr_cfg.get('decoder',lr_cfg['encoder'])},
                ], weight_decay=1e-5)
            else:
                opt=torch.optim.AdamW(model.parameters(), lr=lr_cfg['encoder'], weight_decay=1e-5)

            phase_eps=e-epoch+1; warmup_eps=min(5, phase_eps//4)
            sch=torch.optim.lr_scheduler.SequentialLR(opt, [
                torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_eps),
                torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=phase_eps-warmup_eps),
            ], milestones=[warmup_eps])
            prev_stage=cur_stage

        # 训练
        model.train(); ep={'total':0,'img':0,'lap':0,'struct':0,'vq':0,'ssim':0}
        opt.zero_grad()
        for bi, batch in enumerate(tl):
            projs=batch['projs'].to(device); ct=batch['ct'].to(device)
            mask=batch.get('mask')
            if mask is not None: mask=mask.to(device)
            if projs.dim()!=5: projs=projs.unsqueeze(1) if projs.dim()==4 else projs

            actual=projs.shape[1]; eff=min(cur_views, actual)
            projs=subsample_projections(projs, eff, device) if eff<actual else projs[:,:actual]
            projs_enc=add_angle_encoding(projs, projs.shape[1], device)

            if args.amp:
                with autocast():
                    pred,vq,_=model(projs_enc)
                    ct_a=nn.functional.interpolate(ct, size=pred.shape[2:], mode='trilinear')
                    mask_a=nn.functional.interpolate(mask, size=pred.shape[2:], mode='nearest') if mask is not None else None
                    loss=criterion(pred, ct_a, vq, mask_a)
                scaler.scale(loss['total']/accum).backward()
            else:
                pred,vq,_=model(projs_enc)
                ct_a=nn.functional.interpolate(ct, size=pred.shape[2:], mode='trilinear')
                mask_a=nn.functional.interpolate(mask, size=pred.shape[2:], mode='nearest') if mask is not None else None
                loss=criterion(pred, ct_a, vq, mask_a)
                (loss['total']/accum).backward()

            if (bi+1)%accum==0 or bi==len(tl)-1:
                if args.amp:
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    scaler.step(opt); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                opt.zero_grad()

            for k in ep: v=loss[k]; ep[k]+=(v.item() if isinstance(v,torch.Tensor) else v)
        sch.step()

        # 日志
        nb=len(tl); avg={k:v/nb for k,v in ep.items()}; lr=opt.param_groups[0]['lr']
        print(f'E{epoch:4d} S{cur_stage} lr={lr:.2e} V={cur_views} | '
              f'total={avg["total"]:.4f} img={avg["img"]:.4f} lap={avg["lap"]:.4f} '
              f'struct={avg["struct"]:.4f} vq={avg["vq"]:.4f}')
        tag_map={'total':'总损失 Total Loss','img':'图像损失 Image',
                 'lap':'拉普拉斯损失 Laplacian','struct':'结构损失 Structure',
                 'vq':'码本损失 VQ','ssim':'SSIM'}
        for k in avg: writer.add_scalar(f'Train/{tag_map.get(k,k)}',avg[k],epoch)
        writer.add_scalar('Train/学习率 LR',lr,epoch)
        writer.add_scalar('Train/视角数 n_views',cur_views,epoch)
        writer.add_scalar('Train/阶段 Stage',cur_stage,epoch)

        # 验证
        if epoch%args.eval_every==0 or epoch==total_epochs:
            em=evaluate(model, vl, device, criterion)
            mask_ok = not np.isnan(float(em['psnr_mask']))
            mstr=f' PSNR_mask={em["psnr_mask"]:.2f}dB MAE_mask={em["mae_mask"]:.4f}' if mask_ok else ''
            print(f'  验证 | E{epoch} PSNR={em["psnr"]:.2f}dB{mstr} SSIM={em["ssim"]:.4f} img={em["img"]:.4f}')
            etag={'psnr':'PSNR','psnr_mask':'PSNR (Mask)','mae_mask':'MAE (Mask)',
                  'ssim':'SSIM','img':'图像损失 Image','lap':'拉普拉斯损失 Laplacian',
                  'struct':'结构损失 Structure','total':'总损失 Total Loss'}
            for k in em: writer.add_scalar(f'Eval/{etag.get(k,k)}',em[k],epoch)

            cur_metric=float(em['psnr_mask']) if mask_ok else float(em['psnr'])
            if cur_metric>best_mask_psnr:
                best_mask_psnr=cur_metric; best_epoch=epoch
                torch.save({'epoch':epoch,'model_state':model.state_dict(),
                            'best_mask_psnr':best_mask_psnr,'stage':cur_stage,
                            'views':cur_views,'eval_metrics':em},
                           os.path.join(log_dir,'best_model.pth'))
                print(f'  >> Best (Mask PSNR={best_mask_psnr:.2f}dB)')

        if epoch%args.save_every==0:
            torch.save({'epoch':epoch,'model_state':model.state_dict(),
                        'best_mask_psnr':best_mask_psnr},
                       os.path.join(log_dir,f'ckpt_{epoch:04d}.pth'))

    writer.close()
    print(f'\nDone. Best Mask PSNR: {best_mask_psnr:.2f}dB at epoch {best_epoch} | {log_dir}')


# =========================================================================
# 参数
# =========================================================================

if __name__=='__main__':
    p=argparse.ArgumentParser(description='SparseViewReconstruction v4')
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--vol_size', type=int, nargs=3, default=(128,128,128))
    p.add_argument('--organ', type=str, default='thorax_fast')
    p.add_argument('--train_views', type=int, default=6)
    p.add_argument('--stage1_epochs', type=int, default=200, help='阶段1: 64-view 预训练轮数')
    p.add_argument('--stage1_views', type=int, default=64)
    p.add_argument('--stage2_view_decay', type=str, default='8,4,2',
                   help='阶段2 视角衰减: "8,4,2"=步长模式(从stage1_views递减到train_views), 也可显式列表如"56,48,..."')
    p.add_argument('--stage2_epochs_per_view', type=int, default=40, help='阶段2 每个视角的训练轮数')
    p.add_argument('--stage3_epochs', type=int, default=100, help='阶段3: 6-view 冻结码本微调')
    p.add_argument('--proj_size', type=int, nargs=2, default=(128,128))
    p.add_argument('--n_decoder_ups', type=int, default=1)
    p.add_argument('--epochs', type=int, default=600)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--grad_accum', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--no_amp', action='store_false', dest='amp')
    p.add_argument('--log_dir', type=str, default='./logs')
    p.add_argument('--eval_every', type=int, default=10)
    p.add_argument('--save_every', type=int, default=50)
    args=p.parse_args()
    train(args)
