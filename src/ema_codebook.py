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
        # 通过索引取出量化特征
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
        smoothed = ((self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n)   # 58，59，61，62行都用到了广播
        embed_normalized = self.embed_avg / smoothed.unsqueeze(1)                       # unsqueeze(1)是为了让除法在正确的维度上广播。具体可以问ai
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
        decay=0.99,                     # decay=0.99 是 EMA codebook 的指数移动平均衰减系数，决定码字更新时“保留多少历史信息、吸收多少当前 batch 信息”。
        eps=1e-5,
        kmeans_iters=10,
        kmeans_samples_per_code=4,      # K-means 为每个码字准备的目标采样数量
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
            'kmeans_initialized', torch.tensor(False, dtype=torch.bool)     # 一次性的开关，标记 K-means 初始化是否完成。False 时走样本收集→K-means 流程，True 后直接跳过量化和 EMA 更新。
        )
        # reservoir 大小与码字数成比例，避免从整个三维特征图保存海量向量。
        reservoir_size = self.num_tokens * self.kmeans_samples_per_code     # 预先分配一个固定大小的"蓄水池"，跨多个 batch 收集特征向量，供 K-means 初始化使用。
        # 以下状态注册为 buffer，因此会随 state_dict/checkpoint 完整保存。
        self.register_buffer(
            'kmeans_reservoir',                                             # 为什么需要？ 单个 batch 的特征可能太偏（只覆盖部分分布），跨多个 batch 收集后做 K-means 能得到更具代表性的初始码字。reservoir_size = num_tokens × samples_per_code 确保每个码字平均有 4 个候选样本。
            torch.zeros(reservoir_size, self.codebook_dim),                 
        )
        self.register_buffer(                                               # 作用： 记录 kmeans_reservoir 中实际已写入的样本数。
            'kmeans_reservoir_count', torch.tensor(0, dtype=torch.long)     # 每次 _accumulate_kmeans_reservoir 执行时 count += actual，当 count >= num_tokens 且 batches_seen >= kmeans_init_batches 时触发 K-means。
        )
        self.register_buffer(                                               # 作用： 记录已经采样了多少个 batch。
            'kmeans_batches_seen', torch.tensor(0, dtype=torch.long)        # 两个条件都满足才触发 K-means：1. batches_seen >= kmeans_init_batches（覆盖足够多的 batch，保证多样性）。 2. reservoir_count >= num_tokens（收集了足够多的样本）
        )
        self.register_buffer(                                               # 作用： 记录 EMA 码本已经更新了多少步（即 forward 了多少次）。
            'ema_update_steps', torch.tensor(0, dtype=torch.long)           # 两个用途：1.steps < dead_code_warmup_steps	预热期，不检查死码（让码本先稳定）；2.steps % dead_code_check_interval == 0	每隔 N 步检查一次死码并重置。
        )
        self.register_buffer(                                               # 作用： 累计统计训练以来一共重置了多少个死码，纯监控指标。
            'dead_codes_reinitialized_total', torch.tensor(0, dtype=torch.long) # 每次 _reinitialize_dead_codes 找到死码并替换后，累加：self.dead_codes_reinitialized_total.add_(dead_count)
        )

    @torch.no_grad()
    def _uniform_feature_sample(self, z, sample_count=None):
        """
        Uniformly sample current features without external spatial bias.
        
        z: (B*N, C)  扁平化的特征向量

        是一个高效均匀采样器，从当前 batch 的 N 个特征向量中随机抽取 sample_count 个
        """
        if sample_count is None:
            sample_count = self.num_tokens * self.kmeans_samples_per_code   # 码本中有多少个码字 × 4。这样每个码字平均有 4 个候选样本做 K-means。
        # 不请求超过当前特征总数的样本。
        sample_count = min(z.shape[0], int(sample_count))                   # 上限截断。跨 batch 蓄水时，单批采样数可以小于码字数；只要求最终累计样本数不少于码字数。
        if sample_count <= 0:
            raise ValueError(
                f'feature sampling needs at least one vector, got '
                f'z.shape={tuple(z.shape)} and sample_count={sample_count}'
            )
        if sample_count == z.shape[0]:                                      # 全取，直接返回。如果要采的量恰好等于全量，就不做随机采样，直接返回全部特征。省去无意义的 randint + index_select 开销。
            return z
        # 有放回均匀抽样，避免构造与全部体素数量等长的 randperm。
        indices = torch.randint(                                            # 对于 128³ = 200 万体素的场景，有放回采 2048 个只需要 2048 个随机数，无放回需要先生成 200 万个数的排列再取前 2048 个——开销差了 1000 倍。
            0, z.shape[0], (sample_count,), device=z.device                 # 有放回带来的微小偏差（同一个体素可能被采两次）在 K-means 初始化场景下完全可以忽略，K-means 本身就会迭代收敛，不差这一两个重复样本。
        )
        return z.index_select(0, indices)                                   # torch.index_select(input, dim, index)  是按索引从张量中**选取行（或列）**的操作。dim=0沿第 0 维（行）选取;indices	要取的行的索引列表，如 [3, 0, 7, 3, 1]

    @torch.no_grad()
    def _accumulate_kmeans_reservoir(self, z):
        """
        Collect a fixed sample budget over several shuffled train batches.

        z: (B*N, C)  扁平化的特征向量
        
        背景：为什么要"蓄水"？
        K-means 初始化需要一批有代表性的特征向量作为样本。但单次 forward 的特征分布太窄，所以需要跨多个 batch 采样，凑够一批多样化的样本后再跑 K-means。
        kmeans_reservoir 就是这个"蓄水池"，容量是 num_tokens × kmeans_samples_per_code（如         512×4=2048）。

        """
        capacity = self.kmeans_reservoir.shape[0]           # kmeans_reservoir.shape = (reservoir_size, self.codebook_dim)，而reservoir_size = self.num_tokens * self.kmeans_samples_per_code，即1个码字有4个候选样本
        count = int(self.kmeans_reservoir_count.item())     # kmeans_reservoir_count 作用： 记录 kmeans_reservoir 中实际已写入的样本数
        batches_seen = int(self.kmeans_batches_seen.item()) # kmeans_batches_seen 作用：记录已经采样了多少个 batch
        # 将剩余容量尽量均匀分配给尚未扫描的 batch。
        remaining = capacity - count                                                    # 还剩多少空位
        batches_left = max(1, self.kmeans_init_batches - batches_seen)                  # 还剩几个 batch 可采
        take = min(remaining, max(1, (remaining + batches_left - 1) // batches_left))   # 计算这次该采用多少个
        sampled = self._uniform_feature_sample(z, take)                                 # 从当前 batch 的 N 个体素中随机抽 take 个
        actual = sampled.shape[0]                                                       # 从当前 batch 的 N 个体素中随机抽 take 个
        # copy_ 原位写入 buffer，不建立计算图。
        self.kmeans_reservoir[count:count + actual].copy_(sampled)                      # 追加写入，copy_ 无反向图
        self.kmeans_reservoir_count.add_(actual)                                        # 计数 += 实际写入数
        self.kmeans_batches_seen.add_(1)                                                # batch 数 += 1
        return (
            int(self.kmeans_batches_seen.item()) >= self.kmeans_init_batches            # 采样了足够多的 batch
            and int(self.kmeans_reservoir_count.item()) >= self.num_tokens              # 收集了足够多的样本
        )   # 返回 False → forward返回恒等映射，返回 True → 触发 _initialize_with_kmeans

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
        # “样本数不少于码字数”是启动 K-means 的整体约束，而不是每个
        # reservoir 分片的约束。默认512码字、4样本/码字、8个batch时，
        # 每批只采256个，但累计得到2048个有效样本。
        if samples.shape[0] < self.num_tokens:
            raise ValueError(
                f'K-means needs at least {self.num_tokens} accumulated '
                f'feature vectors, got {samples.shape[0]}'
            )
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
        K = self.num_tokens     # 就是码本中码字的总数

        # 预计算 ||e_j||^2 (所有 chunk 共用)
        w_norm_sq = w_f.pow(2).sum(dim=1)  # (K,)

        # 自适应 chunk size: 限制距离矩阵 ≤ 256 MB
        # d_chunk = (chunk, K) * 4 bytes → chunk * K * 4 ≤ 256 MiB
        max_matrix_bytes = 256 * 1024 * 1024
        chunk_size = max(1, max_matrix_bytes // (K * 4))    # 128³的单样本，体素量有2,097,152个，直接占用4GB导致oom，所以会将4GB分块乘多个256MB块

        encoding_indices_list = []
        enc_sum = torch.zeros(K, device=z_f.device, dtype=torch.float32)
        embed_sum = torch.zeros(K, C, device=z_f.device, dtype=torch.float32)
        # 仅训练模式、阶段允许更新且调用方未禁止时才修改 EMA 状态。
        do_ema = self.training and self.embedding._update and (not no_update)
        if do_ema and not bool(self.kmeans_initialized.item()): # kmeans_initialized：一次性的开关，标记 K-means 初始化是否完成。False 时走样本收集→K-means 流程，True 后直接跳过量化和 EMA 更新。
            ready = self._accumulate_kmeans_reservoir(z_f)
            if not ready:                                               # 还没有收集足够多的数据：状态：模型正在训练，但码本还是随机初始化的垃圾值，不能用于量化。
                '''
                核心设计：在码本未就绪时，不做量化，直接把输入原样返回。这样：

                编码器-解码器的其他部分可以正常前向传播和反向传播（不依赖码本的那部分网络仍然在学习）
                不会因为随机码字产生垃圾量化结果污染梯度
                loss 返回 0，不影响整体训练损失
                '''
                progress = (                                            # 1. 计算初始化进度，如已采样 3/8 个 batch → progress = 0.375。纯监控指标。
                    float(self.kmeans_batches_seen.item())
                    / float(self.kmeans_init_batches)
                )
                zero = z_f.new_zeros(())                                # 2. 创建占位零标量。创建一个与 z_f 同设备同 dtype 的标量 0。所有"暂无意义"的诊断值都填入 0。z_f.new_zeros(()) 保证 Device 一致性（CPU/GPU），比 torch.tensor(0.0) 更安全。
                self._last_diagnostics = {
                    'perplexity': zero.detach(),                        # 还没量化，perplexity 无意义 → 0
                    'normalized_perplexity': zero.detach(),             # 没量化，活跃码字数 → 0
                    'batch_active_codes': zero.detach(),                # 所有码字都算"死"的 → K
                    'batch_active_fraction': zero.detach(),
                    'batch_dead_codes': z_f.new_tensor(
                        float(self.num_tokens)
                    ),
                    'ema_active_codes': zero.detach(),                  # EMA 统计也为空 → 0
                    'ema_active_fraction': zero.detach(),               # 同上 → K
                    'ema_dead_codes': z_f.new_tensor(                   # 同上 → K
                        float(self.num_tokens)
                    ),
                    'dead_codes_reinitialized': zero.detach(),
                    'dead_codes_reinitialized_total': (                 # 但累计值从 buffer 读取（跨 run 持久化）
                        self.dead_codes_reinitialized_total.detach()    
                    ),
                    'kmeans_initialized': self.kmeans_initialized.detach(), # 诚实报告：还没初始化
                    'kmeans_init_progress': z_f.new_tensor(progress),   # 0.0 ~ 1.0，便于 TensorBoard 可视化进度
                }
                # 收集完成前保持恒等映射，避免使用尚未初始化的随机码字。
                return z, zero, zero                                    #    原输入  loss=0  perplexity=0.batch为1-7，恒等映射返回z,0,0。batch=8，则ready=True，执行K-means初始化。Batch=9+，kmeans_initialized=True，正常量化 + EMA 更新。

            # 蓄水池满了，要进行解析了
            # 第 1 行：取出已收集的样本
            samples = self.kmeans_reservoir[                            # kmeans_reservoir：(reservoir_size, C)，如 (2048, 128)	完整水池，尾部可能是未初始化的零。kmeans_reservoir_count：标量，实际写入的样本数。samples	(count, C) 如 (2048, 128)	只取已写入的有效部分
                :int(self.kmeans_reservoir_count.item())                # 为什么不用全量？ 水池预分配了 2048 行，但可能最后一轮蓄水恰好装满，也可能略少于容量。[:count] 精确取出有效数据，尾部零向量不参与 K-means。
            ]
            # 第 2 行：执行 K-means 初始化
            self._initialize_with_kmeans(samples)                       # 之前已详细分析过——用这 2048 个样本跑 10 轮 K-means，聚类出 K 个码字初值，同步写入 weight、cluster_size、embed_avg，并标记 kmeans_initialized = True。
            # 第 3-4 行：刷新码字缓存
            # 但 w_f 变量还指向旧 tensor！（见本函数开头）必须重新读取
            w_f = self.embedding.weight.float()                         # 重新读，指向聚类中心
            w_norm_sq = w_f.pow(2).sum(dim=1)                           # 预计算 ||e_j||²，后续分块距离计算共用

        # 分块查询最近码字，控制峰值显存而不改变最终分配结果。
        # 这是整个 VQ 的核心循环——为每个体素特征向量找到最近的码字，同时累积 EMA 统计。
        for i in range(0, N, chunk_size):
            z_chunk = z_f[i:i + chunk_size]                         # (chunk, C)，如 (131072, 128)。把 N 个特征向量切成若干块，每块最多 [chunk_size]行。循环内的操作互不依赖，结果由 scatter 累加汇总。
            z_norm_sq = z_chunk.pow(2).sum(dim=1, keepdim=True)    # (chunk, 1)，预计算Z^2.对每行（每个特征向量）求模长的平方。keepdim=True 保持 (chunk, 1) 形状，便于后续广播。

            # 分块距离: d = ||z||^2 + ||w||^2 - 2*(z·w)
            d_chunk = z_norm_sq + w_norm_sq - 2 * torch.einsum('bd,nd->bn', z_chunk, w_f)
            #         (chunk,1)   (K,)        (chunk,K)  ← einsum 结果
            #          前2项者均被↓ 广播为 (chunk,K)    einsum是进行批量点积计算z_i * e_j

            idx_chunk = torch.argmin(d_chunk, dim=1)                # (chunk,),每个元素 ∈ [0, K-1].找到最近码字.沿 dim=1（码字维度）取 argmin——每个特征向量归属到距离最近的码字编号。
            encoding_indices_list.append(idx_chunk)                 # 不立即 concat，先 append 到列表等循环结束后一次 [torch.cat]

            # scatter 累加 (EMA 统计和 perplexity, 增量完成)
            enc_sum.scatter_add_(0, idx_chunk, torch.ones_like(idx_chunk, dtype=torch.float32)) # dim=0:沿第 0 维（K 个码字）散布.idx_chunk:每个元素的目标位置（哪个码字）.ones_like:每个元素的值（都是 1）.效果：enc_sum[j] += (idx_chunk 中等于 j 的元素个数)
            if do_ema:                                              # 即统计当前 chunk 中每个码字被命中了几次。
                embed_sum.scatter_add_(0, idx_chunk.unsqueeze(1).expand(-1, C), z_chunk)        # [embed_sum]同理，但累加的是**特征向量本身**而不是 1：
                                                                                                # 步骤	                形状	    含义
                                                                                                # idx_chunk	        (chunk,)    每个向量归属的码字编号
                                                                                                # .unsqueeze(1)	    (chunk, 1)	增加 C 维
                                                                                                # .expand(-1, C)	(chunk, C)	每行复制 C 次，值都是同一个码字编号
                                                                                                # scatter_add_	    —	        embed_sum[j, :] += 所有分配给 j 的 z

        # 拼接分块索引
        encoding_indices = torch.cat(encoding_indices_list, dim=0)  # (N,)。分块循环中每个 chunk 产生了 (chunk_size,) 的索引，现在把它们沿 dim=0 拼接回完整的 (N,)——即 N 个体素各自归属的码字编号。
        # 查表得到量化向量
        z_q = self.embedding(encoding_indices).to(z.dtype)          # (N, C), 回到原精度。self.embedding 是 EmbeddingEMA 实例，其 forward 就是 F.embedding(embed_id, self.weight)——按编号从码本中取出对应的码字向量。结果从 fp32 转回原始 dtype（如 fp16）。
                                                                    # 即每个体素的原始特征被替换为它最近的码字

        # perplexity (fp32, scatter 已累加完成)
        # perplexity 表示当前 batch 实际使用码字分布的有效类别数。
        # 计算概率分布
        avg_probs = enc_sum / enc_sum.sum().clamp(min=1)            # enc_sum[j] = 码字 j 被命中的次数（来自 scatter 累加），除以总命中次数，得到每个码字的使用频率（概率分布），.clamp(min=1) 防止除零
        # 计算 perplexity。
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs.clamp(min=1e-10))))   # 值为1，表示只有1个码字被使用，512表示所有码字完美均匀使用
        # 活跃码字数
        active_codes = (enc_sum > 0).sum()                          # 统计在当前 batch 中至少被命中过一次的码字数量。
        batch_diagnostics = {
            'perplexity': perplexity.detach(),                      # 原始困惑度
            'normalized_perplexity': (                              # 归一化困惑度 ∈ [0, 1]
                perplexity / float(self.num_tokens)
            ).detach(),
            'batch_active_codes': active_codes.detach(),            # 当前 batch 使用了多少码字
            'batch_active_fraction': (                              # 活跃比例 ∈ [0, 1]
                active_codes.float() / float(self.num_tokens)
            ).detach(),
            'batch_dead_codes': (self.num_tokens - active_codes).detach(),  # 当前 batch 中未使用的码字数
        }

        # EMA 更新码本 (训练时, scatter 已累加完成)
        # 更新顺序为计数 → 向量和 → 归一化码字。
        if do_ema:
            self.embedding.cluster_size_ema_update(enc_sum)
            self.embedding.embed_avg_ema_update(embed_sum)
            self.embedding.weight_update(self.num_tokens)
            self.ema_update_steps.add_(1)                           # 记录总步数，供后续死码检查判断预热期和检查间隔。
        reinitialized = 0
        # 预热结束后按固定 EMA forward 间隔检查死码。
        # 两阶段防护：
        # 步数	行为
        # 1-99	预热期，即使有死码也不管（让码本先稳定）
        # 100, 200, 300...	每 100 步扫描一次死码，用当前 batch 特征替换
        if (
            do_ema
            and int(self.ema_update_steps.item()) >= self.dead_code_warmup_steps        # 预热结束（默认 100 步）
            and int(self.ema_update_steps.item()) % self.dead_code_check_interval == 0  # 每 100 步检查一次
        ):  
            reinitialized = self._reinitialize_dead_codes(z_f)                          # _reinitialize_dead_codes 内部判断 cluster_size < 0.1 的码字为死码，从当前特征中随机抽样替换。
        ema_active_codes = (                                                            # 与之前的 batch_active_codes（当前 batch 的瞬时活跃数）不同，这里基于 EMA 的历史平均命中计数。对比两者：
            self.embedding.cluster_size > self.dead_code_threshold  
        ).sum()
        # 指标	                    来源	                含义
        # batch_active_codes	当前 batch 的 enc_sum	瞬时活跃码字数
        # ema_active_codes	    EMA 累积的 cluster_size	历史平均活跃码字数
        # 更能反映码本的长期健康度——一个码字可能当前 batch 没被用到，但 EMA 平均仍然健康。

        # 这段代码做两件事：补全 EMA 级诊断指标，然后保存给外部读取。batch_diagnostics.update(...) — 追加 EMA 级指标，之前 batch_diagnostics 只包含当前 batch 的瞬时统计（perplexity、batch_active_codes 等），这里追加基于 EMA 历史累计的指标：
        batch_diagnostics.update({
            'ema_active_codes': ema_active_codes.detach(),          # EMA 历史平均活跃码字数（cluster_size > 0.1）
            'ema_active_fraction': (
                ema_active_codes.float() / float(self.num_tokens)   # 活跃比例 = ema_active / K，接近 1 才健康
            ).detach(),
            'ema_dead_codes': (
                self.num_tokens - ema_active_codes                  # EMA 历史平均死码数 = K− 活跃数
            ).detach(),
            'dead_codes_reinitialized': torch.tensor(
                reinitialized, device=z_f.device, dtype=torch.float32   # 本轮重置了几个死码（通常是 0）
            ),
            'dead_codes_reinitialized_total': (
                self.dead_codes_reinitialized_total.detach()        # 累计重置了多少死码（跨所有训练步）
            ),
            'kmeans_initialized': self.kmeans_initialized.detach(), # K-means 是否已完成
            'kmeans_init_progress': z_f.new_tensor(1.0),            # 恒为 1.0（能走到这里说明已初始化完毕）
        })
        self._last_diagnostics = batch_diagnostics

        # commitment loss (fp32 计算后转回原精度)
        # commitment loss 只推动编码器特征靠近已选择的码字。
        loss = self.beta * F.mse_loss(z_q.detach().float(), z_f)
        # 关键细节：z_q.detach()——损失只反向传播到 
        # z（编码器输出），不传到码字。码字通过 EMA 更新而非梯度。

        # 谁被推动	        方向
        # 编码器特征 z	 向选定的码字 z_q 靠拢
        # 码字 e_j      通过 EMA 向分配来的特征靠拢（不在计算图中）

        # 这也回答了为什么叫 "commitment loss"——迫使编码器"承诺"靠近选定的码字，而不是让码字来迁就编码器。

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
        self.pre_quant = nn.Conv3d(embedding_dim, embedding_dim, kernel_size=1) # 量化前投影
        self.post_quant = nn.Conv3d(embedding_dim, embedding_dim, kernel_size=1)# 量化后恢复
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

        # 展平为 (B*D*H*W, C)。transpose 只能交换两个特定的维度（一次换一对），而 permute 可以一次性重排所有维度（随意打乱整张顺序）。
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
        """
        Toggle gradient training for pre/post quantization convolutions only.
        
        trainable为true 或 false，用于控制 codebook是否用于训练"""
        for module in (self.pre_quant, self.post_quant):
            for parameter in module.parameters():
                parameter.requires_grad = trainable

    def diagnostics(self):
        return self.codebook.diagnostics()
