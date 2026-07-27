"""
EMA (指数移动平均) 码本 — 从 DeepSparse 项目适配。

核心区别 vs 梯度更新码本:
  - 码本通过 EMA 平滑更新, 而非梯度下降
  - 避免 codebook collapse (所有码字崩塌到少数几个)
  - 标准 VQ-VAE 做法, 码本使用率更均匀

用法:
  cb = EMAVectorQuantizer3D(n_embed=512, embedding_dim=128)
  feat, vq_loss, perplexity = cb(feat_3d)   # feat_3d: (B, C, D, H, W)
"""

# 码本的距离计算、K-means、EMA 统计和直通估计全部由 PyTorch 完成。
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
        # decay 控制历史统计保留比例；eps 防止极小簇除零。
        self.decay = decay
        self.eps = eps
        # K-means 完成前使用随机权重占位，之后会被聚类中心覆盖。
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        # 普通 Python 开关由训练阶段控制，不参与梯度和 EMA 数值保存。
        self._update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        # 当前 batch 的命中次数以 (1-decay) 权重合入历史统计。
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg):
        # 同步维护每个码字所对应特征向量和的 EMA。
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        # 拉普拉斯平滑后的簇大小用于稳定稀有码字的归一化。
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

    def __init__(
        self,
        n_embed,
        embedding_dim,
        beta=1.0,
        decay=0.99,
        eps=1e-5,
        kmeans_iters=10,
        kmeans_samples_per_code=4,
        kmeans_init_batches=8,
        dead_code_threshold=0.1,
        dead_code_check_interval=100,
        dead_code_warmup_steps=100,
    ):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        # beta 只缩放 commitment loss；码字本身不通过优化器更新。
        self.beta = beta
        self.kmeans_iters = int(kmeans_iters)
        self.kmeans_samples_per_code = int(kmeans_samples_per_code)
        self.kmeans_init_batches = int(kmeans_init_batches)
        self.dead_code_threshold = float(dead_code_threshold)
        self.dead_code_check_interval = int(dead_code_check_interval)
        self.dead_code_warmup_steps = int(dead_code_warmup_steps)
        if self.kmeans_iters <= 0:
            raise ValueError('kmeans_iters must be positive')
        if self.kmeans_samples_per_code < 1:
            raise ValueError('kmeans_samples_per_code must be at least 1')
        if self.kmeans_init_batches < 1:
            raise ValueError('kmeans_init_batches must be at least 1')
        if self.dead_code_threshold < 0:
            raise ValueError('dead_code_threshold must be non-negative')
        if self.dead_code_check_interval <= 0:
            raise ValueError('dead_code_check_interval must be positive')
        if self.dead_code_warmup_steps < 0:
            raise ValueError('dead_code_warmup_steps must be non-negative')
        # 实际码字以及 EMA 的计数、向量和都封装在 EmbeddingEMA 中。
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)
        self.register_buffer(
            'kmeans_initialized', torch.tensor(False, dtype=torch.bool)
        )
        # reservoir 大小与码字数成比例，避免从整个三维特征图保存海量向量。
        reservoir_size = self.num_tokens * self.kmeans_samples_per_code
        # 以下状态注册为 buffer，因此会随 state_dict/checkpoint 完整保存。
        self.register_buffer(
            'kmeans_reservoir',
            torch.zeros(reservoir_size, self.codebook_dim),
        )
        self.register_buffer(
            'kmeans_reservoir_count', torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer(
            'kmeans_batches_seen', torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer(
            'ema_update_steps', torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer(
            'dead_codes_reinitialized_total', torch.tensor(0, dtype=torch.long)
        )

    @torch.no_grad()
    def _uniform_feature_sample(self, z, sample_count=None):
        """Uniformly sample current features without external spatial bias."""
        if sample_count is None:
            sample_count = self.num_tokens * self.kmeans_samples_per_code
        # 不请求超过当前特征总数的样本。
        sample_count = min(z.shape[0], int(sample_count))
        if sample_count < self.num_tokens:
            raise ValueError(
                f'K-means needs at least {self.num_tokens} feature vectors, '
                f'got {z.shape[0]}'
            )
        if sample_count == z.shape[0]:
            return z
        # 有放回均匀抽样，避免构造与全部体素数量等长的 randperm。
        indices = torch.randint(
            0, z.shape[0], (sample_count,), device=z.device
        )
        return z.index_select(0, indices)

    @torch.no_grad()
    def _accumulate_kmeans_reservoir(self, z):
        """Collect a fixed sample budget over several shuffled train batches."""
        capacity = self.kmeans_reservoir.shape[0]
        count = int(self.kmeans_reservoir_count.item())
        batches_seen = int(self.kmeans_batches_seen.item())
        # 将剩余容量尽量均匀分配给尚未扫描的 batch。
        remaining = capacity - count
        batches_left = max(1, self.kmeans_init_batches - batches_seen)
        take = min(remaining, max(1, (remaining + batches_left - 1) // batches_left))
        sampled = self._uniform_feature_sample(z, take)
        actual = sampled.shape[0]
        # copy_ 原位写入 buffer，不建立计算图。
        self.kmeans_reservoir[count:count + actual].copy_(sampled)
        self.kmeans_reservoir_count.add_(actual)
        self.kmeans_batches_seen.add_(1)
        return (
            int(self.kmeans_batches_seen.item()) >= self.kmeans_init_batches
            and int(self.kmeans_reservoir_count.item()) >= self.num_tokens
        )

    @staticmethod
    def _nearest_centroid_indices(samples, centroids):
        # 展开平方欧氏距离，避免显式构造 (样本,码字,通道) 三维张量。
        distances = (
            samples.pow(2).sum(dim=1, keepdim=True)
            + centroids.pow(2).sum(dim=1).unsqueeze(0)
            - 2 * samples @ centroids.t()
        )
        return distances.argmin(dim=1), distances

    @torch.no_grad()
    def _initialize_with_kmeans(self, z):
        """Initialize all codewords from a bounded uniform feature sample."""
        samples = self._uniform_feature_sample(z)
        # 从样本中无放回选择初始中心，保证起始码字对应真实特征。
        initial_indices = torch.randperm(
            samples.shape[0], device=samples.device
        )[:self.num_tokens]
        centroids = samples.index_select(0, initial_indices).clone()

        for _ in range(self.kmeans_iters):
            assignments, distances = self._nearest_centroid_indices(
                samples, centroids
            )
            # 统计每个中心的样本数，并以 scatter_add 累积向量和。
            counts = torch.bincount(
                assignments, minlength=self.num_tokens
            ).to(samples.dtype)
            sums = torch.zeros_like(centroids)
            sums.scatter_add_(
                0,
                assignments.unsqueeze(1).expand(-1, self.codebook_dim),
                samples,
            )
            nonempty = counts > 0
            centroids[nonempty] = (
                sums[nonempty] / counts[nonempty].unsqueeze(1)
            )
            # 空簇使用当前误差最大的样本重置，避免初始化后立即产生死码。
            empty_count = int((~nonempty).sum().item())
            if empty_count:
                nearest_distances = distances.gather(
                    1, assignments.unsqueeze(1)
                ).squeeze(1)
                replacement_indices = nearest_distances.topk(
                    min(empty_count, samples.shape[0])
                ).indices
                replacements = samples.index_select(
                    0, replacement_indices
                )
                if replacements.shape[0] < empty_count:
                    repeat = (
                        empty_count + replacements.shape[0] - 1
                    ) // replacements.shape[0]
                    replacements = replacements.repeat(repeat, 1)
                centroids[~nonempty] = replacements[:empty_count]

        assignments, _ = self._nearest_centroid_indices(samples, centroids)
        counts = torch.bincount(
            assignments, minlength=self.num_tokens
        ).to(samples.dtype)
        # 聚类结束后同时初始化码字、簇计数和向量和，保持 EMA 三者一致。
        self.embedding.weight.data.copy_(centroids)
        self.embedding.cluster_size.data.copy_(counts)
        self.embedding.embed_avg.data.copy_(
            centroids * counts.unsqueeze(1)
        )
        self.kmeans_initialized.fill_(True)
        # 初始化完成后释放 reservoir 中的旧特征内容，但保留固定 buffer 形状。
        self.kmeans_reservoir.zero_()

    @torch.no_grad()
    def _reinitialize_dead_codes(self, z):
        """Replace persistently unused codewords with uniform current features."""
        # 只有 EMA 平均命中数低于阈值的码字才被视为死亡。
        dead = self.embedding.cluster_size < self.dead_code_threshold
        dead_count = int(dead.sum().item())
        if dead_count == 0:
            return 0
        indices = torch.randint(0, z.shape[0], (dead_count,), device=z.device)
        # 从当前 batch 的真实特征中抽取替代向量，使新码字位于当前分布上。
        replacements = z.index_select(0, indices)
        reset_size = max(
            self.dead_code_threshold,
            float(self.embedding.eps),
        )
        self.embedding.weight.data[dead] = replacements
        self.embedding.cluster_size.data[dead] = reset_size
        self.embedding.embed_avg.data[dead] = replacements * reset_size
        self.dead_codes_reinitialized_total.add_(dead_count)
        return dead_count

    def forward(self, z, no_update=False):
        """
        z: (B*N, C)  扁平化的特征向量
        返回: z_q (量化后), loss, perplexity
        使用分块计算避免 O(N×K) 距离矩阵 OOM
        """
        # 强制 fp32 计算距离，防止 AMP fp16 溢出
        # 即使外部处于 AMP，距离和 EMA 累加也固定使用 fp32。
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
        # 仅训练模式、阶段允许更新且调用方未禁止时才修改 EMA 状态。
        do_ema = self.training and self.embedding._update and (not no_update)
        if do_ema and not bool(self.kmeans_initialized.item()):
            ready = self._accumulate_kmeans_reservoir(z_f)
            if not ready:
                progress = (
                    float(self.kmeans_batches_seen.item())
                    / float(self.kmeans_init_batches)
                )
                zero = z_f.new_zeros(())
                self._last_diagnostics = {
                    'perplexity': zero.detach(),
                    'normalized_perplexity': zero.detach(),
                    'batch_active_codes': zero.detach(),
                    'batch_active_fraction': zero.detach(),
                    'batch_dead_codes': z_f.new_tensor(
                        float(self.num_tokens)
                    ),
                    'ema_active_codes': zero.detach(),
                    'ema_active_fraction': zero.detach(),
                    'ema_dead_codes': z_f.new_tensor(
                        float(self.num_tokens)
                    ),
                    'dead_codes_reinitialized': zero.detach(),
                    'dead_codes_reinitialized_total': (
                        self.dead_codes_reinitialized_total.detach()
                    ),
                    'kmeans_initialized': self.kmeans_initialized.detach(),
                    'kmeans_init_progress': z_f.new_tensor(progress),
                }
                # 收集完成前保持恒等映射，避免使用尚未初始化的随机码字。
                return z, zero, zero
            samples = self.kmeans_reservoir[
                :int(self.kmeans_reservoir_count.item())
            ]
            self._initialize_with_kmeans(samples)
            w_f = self.embedding.weight.float()
            w_norm_sq = w_f.pow(2).sum(dim=1)

        # 分块查询最近码字，控制峰值显存而不改变最终分配结果。
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
        # perplexity 表示当前 batch 实际使用码字分布的有效类别数。
        avg_probs = enc_sum / enc_sum.sum().clamp(min=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs.clamp(min=1e-10))))
        active_codes = (enc_sum > 0).sum()
        batch_diagnostics = {
            'perplexity': perplexity.detach(),
            'normalized_perplexity': (
                perplexity / float(self.num_tokens)
            ).detach(),
            'batch_active_codes': active_codes.detach(),
            'batch_active_fraction': (
                active_codes.float() / float(self.num_tokens)
            ).detach(),
            'batch_dead_codes': (self.num_tokens - active_codes).detach(),
        }

        # EMA 更新码本 (训练时, scatter 已累加完成)
        # 更新顺序为计数 → 向量和 → 归一化码字。
        if do_ema:
            self.embedding.cluster_size_ema_update(enc_sum)
            self.embedding.embed_avg_ema_update(embed_sum)
            self.embedding.weight_update(self.num_tokens)
            self.ema_update_steps.add_(1)
        reinitialized = 0
        # 预热结束后按固定 EMA forward 间隔检查死码。
        if (
            do_ema
            and int(self.ema_update_steps.item()) >= self.dead_code_warmup_steps
            and int(self.ema_update_steps.item())
                % self.dead_code_check_interval == 0
        ):
            reinitialized = self._reinitialize_dead_codes(z_f)
        ema_active_codes = (
            self.embedding.cluster_size > self.dead_code_threshold
        ).sum()
        batch_diagnostics.update({
            'ema_active_codes': ema_active_codes.detach(),
            'ema_active_fraction': (
                ema_active_codes.float() / float(self.num_tokens)
            ).detach(),
            'ema_dead_codes': (
                self.num_tokens - ema_active_codes
            ).detach(),
            'dead_codes_reinitialized': torch.tensor(
                reinitialized, device=z_f.device, dtype=torch.float32
            ),
            'dead_codes_reinitialized_total': (
                self.dead_codes_reinitialized_total.detach()
            ),
            'kmeans_initialized': self.kmeans_initialized.detach(),
            'kmeans_init_progress': z_f.new_tensor(1.0),
        })
        self._last_diagnostics = batch_diagnostics

        # commitment loss (fp32 计算后转回原精度)
        # commitment loss 只推动编码器特征靠近已选择的码字。
        loss = self.beta * F.mse_loss(z_q.detach().float(), z_f)

        # straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, loss, perplexity

    def freeze(self):
        # 冻结仅停止 EMA 数值更新，量化查询仍正常工作。
        self.embedding._update = False

    def unfreeze(self):
        self.embedding._update = True

    def diagnostics(self):
        """Return diagnostics from the most recent forward pass."""
        return getattr(self, '_last_diagnostics', {})


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

    def __init__(
        self,
        n_embed,
        embedding_dim,
        beta=1.0,
        decay=0.99,
        kmeans_iters=10,
        kmeans_samples_per_code=4,
        kmeans_init_batches=8,
        dead_code_threshold=0.1,
        dead_code_check_interval=100,
        dead_code_warmup_steps=100,
    ):
        super().__init__()
        self.pre_quant = nn.Conv3d(embedding_dim, embedding_dim, kernel_size=1)
        self.post_quant = nn.Conv3d(embedding_dim, embedding_dim, kernel_size=1)
        self.codebook = EMAVectorQuantizer(
            n_embed=n_embed, embedding_dim=embedding_dim,
            beta=beta, decay=decay,
            kmeans_iters=kmeans_iters,
            kmeans_samples_per_code=kmeans_samples_per_code,
            kmeans_init_batches=kmeans_init_batches,
            dead_code_threshold=dead_code_threshold,
            dead_code_check_interval=dead_code_check_interval,
            dead_code_warmup_steps=dead_code_warmup_steps,
        )

    def forward(self, x, no_update=False):
        """
        x: (B, C, D, H, W)
        返回: x_q (量化后), vq_loss, perplexity
        """
        # 保存 batch/通道信息，以便量化后恢复原三维网格。
        B, C = x.shape[:2]
        # pre_quant 学习把上游特征变换到更适合码字查询的空间。
        x = self.pre_quant(x)

        # 展平为 (B*D*H*W, C)
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, C)

        x_q_flat, vq_loss, perplexity = self.codebook(x_flat, no_update=no_update)

        # 恢复形状
        x_q = x_q_flat.reshape(B, *x.shape[2:], C).permute(0, 4, 1, 2, 3)
        # post_quant 将离散码字特征重新适配给下游解码器。
        x_q = self.post_quant(x_q)

        return x_q, vq_loss, perplexity

    def freeze(self):
        self.codebook.freeze()
        print('[EMA Codebook] Frozen.')

    def unfreeze(self):
        self.codebook.unfreeze()
        print('[EMA Codebook] Unfrozen.')

    def set_adapter_trainable(self, trainable):
        """Toggle gradient training for pre/post quantization convolutions only."""
        for module in (self.pre_quant, self.post_quant):
            for parameter in module.parameters():
                parameter.requires_grad = trainable

    def diagnostics(self):
        return self.codebook.diagnostics()
