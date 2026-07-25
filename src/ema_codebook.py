"""
EMA (指数移动平均) 码本 — 从 DeepSparse 项目适配。

核心区别 vs 梯度更新码本:
  - 码本通过 EMA 平滑更新, 而非梯度下降
  - 避免 codebook collapse (所有码字崩塌到少数几个)
  - 标准 VQ-VAE 做法, 码本使用率更均匀

用法:
  cb = EMAVectorQuantizer3D(n_embed=1024, embedding_dim=128)
  feat, vq_loss, perplexity = cb(feat_3d)   # feat_3d: (B, C, D, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# EMA Embedding: 维护一个可查询的 codebook，用 EMA 统计来更新
# =========================================================================

class EmbeddingEMA(nn.Module):
    """
    self.weight:    最终的 codebook 矩阵 [num_tokens, codebook_dim]
    self.cluster_size: 每个码字被选中的频次 EMA 统计 [num_tokens]
    self.embed_avg: 每个码字对应输入向量的 EMA 累计和 [num_tokens, codebook_dim]
    """
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        self._update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed = ((self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n)
        embed_normalized = self.embed_avg / smoothed.unsqueeze(1)
        # 只更新被使用过的码字, 未使用的保持原值 (防止除以 eps 导致数值爆炸)
        used = self.cluster_size > 0
        self.weight.data[used] = embed_normalized[used]


# =========================================================================
# EMA Vector Quantizer (独立于 2D/3D)
# =========================================================================

class EMAVectorQuantizer(nn.Module):
    """EMA 更新的 vector quantization。把输入特征映射到最近的 codebook 向量。"""

    def __init__(self, n_embed, embedding_dim, beta=1.0, decay=0.99, eps=1e-5):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

    def forward(self, z, no_update=False):
        """
        z: (B*N, C)  扁平化的特征向量
        返回: z_q (量化后), loss, perplexity
        使用分块计算避免 O(N×K) 距离矩阵 OOM
        """
        # 强制 fp32 计算距离，防止 AMP fp16 溢出
        z_f = z.float()
        w_f = self.embedding.weight.float()
        N, C = z_f.shape
        K = self.num_tokens

        # 预计算 ||e_j||^2 (所有 chunk 共用)
        w_norm_sq = w_f.pow(2).sum(dim=1)  # (K,)

        # 自适应 chunk size: 限制距离矩阵 ≤ 256 MB
        # d_chunk = (chunk, K) * 4 bytes → chunk * K * 4 ≤ 256 MiB
        max_matrix_bytes = 256 * 1024 * 1024
        chunk_size = max(1, max_matrix_bytes // (K * 4))

        encoding_indices_list = []
        enc_sum = torch.zeros(K, device=z_f.device, dtype=torch.float32)
        embed_sum = torch.zeros(K, C, device=z_f.device, dtype=torch.float32)
        do_ema = self.training and self.embedding._update and (not no_update)

        for i in range(0, N, chunk_size):
            z_chunk = z_f[i:i + chunk_size]                         # (chunk, C)
            z_norm_sq = z_chunk.pow(2).sum(dim=1, keepdim=True)    # (chunk, 1)

            # 分块距离: d = ||z||^2 + ||w||^2 - 2*(z·w)
            d_chunk = z_norm_sq + w_norm_sq - 2 * torch.einsum('bd,nd->bn', z_chunk, w_f)

            idx_chunk = torch.argmin(d_chunk, dim=1)                # (chunk,)
            encoding_indices_list.append(idx_chunk)

            # scatter 累加 (EMA 统计和 perplexity, 增量完成)
            enc_sum.scatter_add_(0, idx_chunk, torch.ones_like(idx_chunk, dtype=torch.float32))
            if do_ema:
                embed_sum.scatter_add_(0, idx_chunk.unsqueeze(1).expand(-1, C), z_chunk)

        encoding_indices = torch.cat(encoding_indices_list, dim=0)  # (N,)
        z_q = self.embedding(encoding_indices).to(z.dtype)          # (N, C), 回到原精度

        # perplexity (fp32, scatter 已累加完成)
        avg_probs = enc_sum / enc_sum.sum().clamp(min=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs.clamp(min=1e-10))))

        # EMA 更新码本 (训练时, scatter 已累加完成)
        if do_ema:
            self.embedding.cluster_size_ema_update(enc_sum)
            self.embedding.embed_avg_ema_update(embed_sum)
            self.embedding.weight_update(self.num_tokens)

        # commitment loss (fp32 计算后转回原精度)
        loss = self.beta * F.mse_loss(z_q.detach().float(), z_f)

        # straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, loss, perplexity

    def freeze(self):
        self.embedding._update = False

    def unfreeze(self):
        self.embedding._update = True


# =========================================================================
# 3D 封装 (适配当前项目)
# =========================================================================

class EMAVectorQuantizer3D(nn.Module):
    """
    封装 EMA VQ 用于 3D 特征图。
    pre_quant:  量化前投影 (1×1×1 Conv)
    codebook:   EMA 码本
    post_quant: 量化后恢复 (1×1×1 Conv)
    """

    def __init__(self, n_embed, embedding_dim, beta=1.0, decay=0.99):
        super().__init__()
        self.pre_quant = nn.Conv3d(embedding_dim, embedding_dim, kernel_size=1)
        self.post_quant = nn.Conv3d(embedding_dim, embedding_dim, kernel_size=1)
        self.codebook = EMAVectorQuantizer(
            n_embed=n_embed, embedding_dim=embedding_dim,
            beta=beta, decay=decay,
        )

    def forward(self, x, no_update=False):
        """
        x: (B, C, D, H, W)
        返回: x_q (量化后), vq_loss, perplexity
        """
        B, C = x.shape[:2]
        x = self.pre_quant(x)

        # 展平为 (B*D*H*W, C)
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, C)

        x_q_flat, vq_loss, perplexity = self.codebook(x_flat, no_update=no_update)

        # 恢复形状
        x_q = x_q_flat.reshape(B, *x.shape[2:], C).permute(0, 4, 1, 2, 3)
        x_q = self.post_quant(x_q)

        return x_q, vq_loss, perplexity

    def freeze(self):
        self.codebook.freeze()
        print('[EMA Codebook] Frozen.')

    def unfreeze(self):
        self.codebook.unfreeze()
        print('[EMA Codebook] Unfrozen.')
