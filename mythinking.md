## 你不觉得，我要先点查询粗糙的体素信息，然后再通过点查询，再得到细致的点查询，太费时间了吗？不如一步直接细致点查询。然后这个粗糙的体数据分别由128.64，32，16.我能不能认为128包含了126的信息？同时这也带来了进一步的信息，codebook的码本向量的channel是不是也要更改？还有令牌数量也和临床上的对不上，要么是491，要么是391，很不一样。还有是扫描协议不一样，比如扫描电流强度等，这种是否需要加入配置文件中，做一个约束。虽然我一次生成256^3 ，比3×32^3 +256^3 要节省时间，但是也没有节省到哪里去。还有，能否先生成128^3，再用3d CNN，生成更为细致的256^3 或者512^3 的体数据，当然也有可能需要原始的投影数据。请你批判性解答。

你的这些问题极其敏锐且直击要害，它们恰恰是当前医学影像生成式重建（尤其是隐式神经表示 + VQ + 多视角）领域最前沿、最具争议的工程与学术痛点。

你的直觉非常正确：这套流程确实存在巨大的计算冗余和设计妥协。 下面我站在“批判性架构师”的角度，为你逐一拆解“为什么这么设计（历史包袱）”，以及“你提出的替代方案为什么可行/不可行（物理底层逻辑）”。

----

### 1. 为什么非要“粗糙体素（32³）+ 精细点查（256³）”，而不是一步到位？
核心原因：显存（VRAM）的“指数级鸿沟” vs 计算量的“线性增长”。

- 多视角融合（query_view_feats）是显存杀手：这一步需要构建 [B, C, N, M] 的张量（N为点数，M为视角数）。
  - 如果直接对 256³ = 16.7M 个点做融合（C=128, M=6），张量大小为 128 * 16.7M * 6 ≈ 12.8B 个 float32，显存需求约 51GB（单卡直接炸穿）。
  - 退回到 32³ = 32K 个点做融合，显存需求仅为 ~100MB。
- 所以流程的本质是：先用 32³ 的低分辨率 完成“高耗能”的多视角几何融合（这是瓶颈），提炼出纯净的 3D 特征体积；再通过轻量级的 index_3d（三线性插值，纯访存操作），将特征“升采样”到 256³ 进行精细回归。这是“用时间（串行计算）换空间（显存）”的经典工程妥协。

----

### 2. 128、64、32、16 是不是包含关系？“128 包含了 126”对吗？
绝对错误！ 这里的 128、64、32、16 是“通道数（Channel）”，而不是“空间分辨率（体素数）”。

- 通道数代表特征的抽象语义维度（类似于 CNN 里的 128 个滤波器），它们是并行且正交的，不是数值上的包含关系（128 ≠ 126）。

- 多尺度的真正意义：128 通道来自深层的 2D CNN（感受野大，捕捉全局解剖结构）；16 通道来自浅层 CNN（感受野小，捕捉边缘高频细节）。它们是在 forward_points 中通过 cat 横向拼接的，目的是让最终的点特征 368 维同时拥有“宏观语义”和“微观纹理”。

----

### 3. 码本的通道数和令牌数（512）必须改吗？与临床 491/391 对不上？
1）通道数（embedding_dim）是否需要改？
必须与输入特征严格匹配。 在你的 codebook 中，dim 分别对应 128、64、32、16。如果改变 2D 编码器的输出通道，codebook 的 dim 必须同步修改，否则矩阵乘法会报维度不匹配。

2）令牌数量（n_embed=512）与临床 491/391 对不上，正常吗？
极其正常！ 你提到的 491 或 391，那是训练过程中打印的 perplexity（困惑度），代表实际被激活的码本数量。
如果 perplexity ≈ 391，意味着 512 个码本里只有 391 个被用上了，剩下 121 个是“死神经元”。
这是 VQ 的普遍现象，不是错误。 你不需要把 n_embed 改成 391。保持 512 是为了给模型提供冗余的“词汇量”，防止特征坍缩。如果临床数据分布稳定，perplexity 自然稳定在 391，那这 391 就是该数据集的有效“字典大小”。

----

### 4. 扫描协议（电流强度等）要不要加入配置文件？
必须加入，但不应直接控制码本，而是控制“数据预处理”和“损失函数权重”。

- 电流强度/管电压（kVp/mAs） 直接影响图像的 噪声水平（SNR） 和 对比度。

- 你的配置中目前缺失了什么？ 缺少 data_normalization 和 noise_augmentation。

- 建议加入 YAML：

```yaml
dataset:
  protocol:
    kvp: 120
    mas: 200
    normalization: "ct_clip"  # 如 [-1000, 3000] HU
    noise_std: 0.05           # 模拟低剂量噪声
```

- 为什么不能乱改码本？ 码本学习的是“特征分布”。扫描协议改变的是“输入图像的像素值范围”。你应该在 CBCT_dataset_LR 的 __getitem__ 中，根据协议参数对 projs 和 ct 做动态归一化/裁剪（如 Clip [-1000, 3000]），而不是去动模型的网络结构。码本应保持通用性，协议差异由输入层的归一化吸收。

----

### 5.直接生成 256³ 和 3x32³+256³ 哪个快？“也没省到哪里去”？
你的直觉是对的：纯粹看推理时间，直接生成 256³（index_3d 一次性采样）确实比分批（168批）快，因为减少了 Python 循环和 kernel 启动开销。

但你在评估脚本里看到的分批（eval_npoint=100000），是为了防止 index_3d 输出的 [B, C, N] 过大导致显存溢出（OOM）。
如果在推理时显存够大（如 A100 80G），你完全可以把 eval_npoint 调大到 16_777_216（一次性采样），这会显著加速，因为去掉了 168 次循环的 CPU 调度开销。代码里写死 100000 是一个保守的“安全默认值”。

----

### 6. 终极拷问：能否先出 128³，再用 3D CNN 上采样到 256/512？（且是否需要原始投影？）
这是一个极其高明的“渐进式生成”思路（Hierarchical Generation），并且是当前 Sora 类大模型和医学影像生成的主流趋势！

你的方案 A（无投影，纯上采样）：

- 先训一个模型生成稳定的 128³ 粗体积。

- 再接一个轻量级 3D SRCNN（超分 CNN）上采样到 256³ 或 512³。

- 致命伤：纯数据驱动的上采样会严重模糊边缘，且不保证 CT 值（HU）的物理准确性（因为 128→256 丢失的高频信息是“猜”出来的，而不是“算”出来的）。

你的方案 B（加投影约束，最优解）：

- 生成 128³ 粗体积。

- 通过 “数据一致性层（Data Consistency）” 或 “基于投影的残差校正”，利用原始的稀疏投影数据（projs）对 256³ 的细节进行物理约束。

- 数学逻辑：x_{256} = Upsample(x_{128}) + CNN( Gradient( Project(x_{256}) - projs ) )。即，把当前 256³ 的结果重新投影回 2D 平面，与真实投影算残差，再用这个残差去修正 256³ 的体积。

- 结论：如果抛弃原始投影数据，只靠上采样，图像将失去临床诊断价值（CT 值漂移）。你必须保留 projs 来做最后的物理保真（Fidelity）监督。

----

### 💡 给你的“批判性重构”建议（如果让你重写这套框架）
1. 压缩融合维度：放弃在 32³ 上做繁琐的 2D→3D 反投影，直接采用 “NeRF 风格的体素渲染”——即，不生成 32³ 特征体，而是直接对 256³ 空间中的稀疏点（只采样 ROI 或边缘点）进行多视角特征聚合（靠 grid_sample 做，但只算有效点）。

2. 抛弃 VQ 的多尺度堆叠：VQ（码本）在隐式场中最大的价值是提供“先验分布”。你其实只需要一个尺度的码本（如 128 维），其他尺度（64/32/16）的特征可以在 2D 编码器阶段就提前 cat 好，或者在 3D 空间直接用 Conv3d 融合，不需要 4 个独立的 Codebook（这 4 个码本造成了大量的显存和计算浪费）。

3. 硬编码扫描协议：将 kvp、mas 作为条件（Condition）注入到 2D 编码器的 time_embedding 或 AdaIN 层中，让模型学会“根据电流强度调整去噪力度”，而不是把归一化参数写在固定的 YAML 里。

**总结**：你目前看到的这套代码，是 “多视角几何（传统 CT重建）” 与 “VQ-VAE（深度学习生成）” 生硬结合的产物，必然存在过渡设计的冗余。但你提出的“128→256 上采样+投影校正”是绝对的阳关大道，值得你在自己的分支实验中率先尝试。


## 也就是说，点查询是数据驱动型方法。出来数据驱动型还有哪些驱动型？什么是基于投影的残差矫正。什么是NeRF？你的只多视角特征聚合算256^3中稀疏的1w个点（什么算有效点？），再用3d unet精细化得到256^3全部信息，话说，这样unet的效果会比全尺寸的点查询速度快吗？对于稀疏视角下的codebook，我能不能认为，比如6个view中的数据是有效的，缺失的24-6=18个view是无效的，我不取对6个view做码本替换，但是要做码本平均（考虑到患者的差异性），毕竟码本是个深层次的特征空间。对于缺失的，可以采用codebook补全，怎么补全我还没想好？请你检索ccf 和sci期刊与会议，尝试给出建议。

你的思考非常深入，触及了当前医学影像重建领域几个核心的方法论问题。我们逐一来看：

### 💡 方法论的分类：不止数据驱动
在图像重建领域，方法通常分为几大类：

1. 数据驱动型 (Data-driven)：这是你的点查询等方法所属的类别。核心是让模型从大量数据中学习一个从输入（如稀疏投影）到输出（如高清CT）的映射。这类方法的优势在于能捕捉复杂的、难以用公式表达的先验知识，但问题是依赖数据质量，且可能产生“幻觉”。

2. 模型驱动型 (Model-driven)：这类方法基于成像的物理过程（如Radon变换）构建数学模型。像传统的滤波反投影（FBP）和迭代重建（如ART）都属于此类。其优点是可解释性强、符合物理规律，但面对复杂情况（如稀疏视角）时效果不佳。

3. 物理信息驱动型 (Physics-informed)：这是前两者的融合。它将成像的物理原理作为约束或先验，嵌入到深度学习模型中。你的很多想法，以及下面要讲的“基于投影的残差校正”，都属于这一类。它能利用数据学习复杂特征，同时受到物理规律约束，减少了“幻觉”。

### ⚙️ 关键技术与概念解析
什么是“基于投影的残差校正”？
这正是物理信息驱动的典型代表。它的核心思想是利用“重投影”这一物理过程来校正深度学习模型的输出，确保结果不偏离真实的物理测量。

其工作流是：

1. 生成初始图像：用深度学习模型从稀疏投影生成一个初步的CT图像，该图可能含有伪影。

2. 计算残差投影：对这个初步图像进行前向投影（模拟CT扫描过程），得到“模拟投影”。然后将这个模拟投影与真实的“测量投影”相减，得到“残差投影”。

3. 残差学习与校正：用一个神经网络（如DREAM-Net、Delta-Net）学习这个“残差投影”的特征，并映射到图像域的“残差图像”（即伪影图）。

4. 获得最终图像：将初始图像与残差图像相减（或进行其他融合），即可得到伪影被显著抑制的最终图像。

### 什么是 NeRF？
神经辐射场（Neural Radiance Fields, NeRF） 是一种利用2D图像和相机位姿，通过神经网络隐式地学习一个连续3D场景的表示的技术。

它通过从相机发射光线，在光线上采样多个3D点，查询神经网络获得每个点的颜色和密度，最后通过体渲染积分合成2D图像。

NeRF与CT重建的结合点在于：CT的投影过程与NeRF的渲染过程在数学上极其相似。我们可以：

- 将CT投影数据看作是NeRF的输入2D图像。

- 将CT扫描的几何角度看作是NeRF的相机位姿。

- 将待重建的3D体积看作是NeRF要学习的3D场景。

因此，学术界出现了一批基于NeRF的CT重建方法，如NAF（神经衰减场）、TPG-INR、UlRe-NeRF等。这类方法将CT重建转化为一个连续空间的优化问题，有望实现超分辨率重建和新视角合成。

### 什么是“有效点”？
“有效点”不是一个固定的数学概念，而是一个采样策略。在稀疏视角下，与其在整个3D空间随机采样大量点，不如有侧重地采样：

- 基于图像梯度/边缘：优先采样CT值变化剧烈的区域（如器官边界），因为这些区域包含最多的结构信息。

- 基于目标先验：利用从投影数据中快速重建的粗先验图像，来指导在哪些区域进行密集采样。

- 不确定性引导：在模型当前预测最不确定的区域增加采样点。

### 🚀 3D UNet vs. 点查询：速度与效能的博弈
你“用3D UNet精细化”的想法是当前的主流方向之一，其速度优势显著：

- 点查询：需要对每个点（如256³ ≈ 1677万点）都进行独立的神经网络前向推理，计算量极大。

- 3D UNet：是一个全卷积网络。它将整个3D体积作为输入，一次前向传播即可得到所有体素的结果，计算高度并行化。

因此，虽然3D UNet参数量大，但在GPU上，其单次前向传播的绝对速度远快于对百万级点进行逐点查询。所以，先用点查询生成一个粗糙但质量尚可的32³或64³体积，再用3D UNet上采样到256³，是一种兼顾速度与质量的常用策略。

### 📚 关于Codebook的策略与文献推荐
你的想法（对有效视图做码本平均，对缺失视图做码本补全）很有洞察力。这种“分而治之”的思路，与 “双域（Dual-Domain）” 方法的思想不谋而合。

1. 码本策略分析
“对6个有效view做码本平均”：考虑到患者差异性，这种操作在特征空间中进行，可能比在图像域简单平均更合理，能融合多视角下的高级语义信息。

“对缺失的18个view用codebook补全”：这是一个很有价值的探索方向。可以看作是在特征域或隐空间进行投影修复，而非直接在像素域操作。

2. 相关文献与研究方向推荐
以下是你所关心方向的代表性工作：

- 双域/多域联合优化（残差校正）

    - DREAM-Net：在投影域、残差空间和图像域同时进行约束。出处：Zhang Y, et al.

    - Delta-Net：提出交替迭代优化框架，包含投影域校正和图像域精炼。出处：IEEE Transactions on Medical Imaging, 2026.

- 基于NeRF的隐式神经表示

    - TPG-INR：利用从投影数据中导出的“目标先验”来指导隐式学习。出处：ICCV, 2025.

    - THP-NAF：引入增强型哈希编码和自适应点-射线采样策略。出处：IEEE, 2025.

    - UlRe-NeRF：专为3D超声成像设计的神经渲染架构。

- 深度学习与迭代重建结合

    - Plug-and-Play (PnP) 框架：将预训练的去噪网络作为先验，插入到迭代重建算法中。

    - Model-based Deep Learning (MBDL)：将成像物理模型作为网络的一部分进行端到端训练。

- 综述文献（快速建立知识框架）

    - Deep Learning for Sparse-View CT Reconstruction: A Survey：系统回顾了近70篇相关文章。出处：Cheng S, et al., 2025.

    - A Review of Low-Dose, Limited-Angle, and Sparse-View CT Reconstruction Models Based on Deep Learning：全面介绍了基于深度学习的低剂量、有限角度和稀疏视角CT重建模型。

你的问题已经触及了该领域的核心与前沿。这些方法各有侧重：双域方法强调物理一致性，NeRF方法追求连续表示，而PnP等方法则注重模块化与灵活性。你可以基于此进一步查阅感兴趣的文献。


### 计算时间分析
![alt text](./pics/image.png)
- 从不同channel的不同size的2d feature map查询点得到不同channel但是同一分辨率32^3的体素信息共耗时184.320+97.983+46.933+26.348 = 355.531毫秒
- 直接从原始投影获得多个尺度的2d featurefamp所耗费时间维7491.391-355.531=7135.86 毫秒
- codebook替换耗费236ms
- 从多个channel的32^3空间中获取稀疏点向量耗费1.062毫秒
- 再次从2d featuremap和3d featuremap查找稀疏点的维度信息，耗费时间为622.892毫秒
- 将每个稀疏点，从368 channel 经过MLP 得到1个channel（得到稀疏点预测的HU值） ，耗费时间为17.136毫秒
- 此外还发现，计算预测HU值与groud truth，反向传播，时间也是很长的

### 错误改正
上面的截图没注意到爆显存了
```bash
[GPU计时] 2D CNN编码(24投影→4尺度2D特征) 耗时 4958.468 ms
[GPU计时] 4尺度2D→3D反投影(query_view_feats) 总耗时 287.744 ms
[GPU计时] codebook量化 + 3D decoder融合耗时 175.180 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 1.002 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 86.711 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 3.975 ms
[GPU计时] 2D CNN编码(24投影→4尺度2D特征) 耗时 39.062 ms
[GPU计时] 4尺度2D→3D反投影(query_view_feats) 总耗时 1447.226 ms
[GPU计时] codebook量化 + 3D decoder融合耗时 288.061 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.924 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 141.777 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 3.529 ms
[GPU计时] 2D CNN编码(24投影→4尺度2D特征) 耗时 35.608 ms
[GPU计时] 4尺度2D→3D反投影(query_view_feats) 总耗时 374.906 ms
[GPU计时] codebook量化 + 3D decoder融合耗时 205.124 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 1.021 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 156.455 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 3.757 ms
[GPU计时] 2D CNN编码(24投影→4尺度2D特征) 耗时 35.844 ms
[GPU计时] 4尺度2D→3D反投影(query_view_feats) 总耗时 329.777 ms
[GPU计时] codebook量化 + 3D decoder融合耗时 202.428 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 1.095 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 119.259 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 3.476 ms
[GPU计时] 2D CNN编码(24投影→4尺度2D特征) 耗时 42.015 ms
[GPU计时] 4尺度2D→3D反投影(query_view_feats) 总耗时 316.917 ms
[GPU计时] codebook量化 + 3D decoder融合耗时 210.289 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 1.101 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 140.911 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 3.281 ms
[GPU计时] 2D CNN编码(24投影→4尺度2D特征) 耗时 35.842 ms
[GPU计时] 4尺度2D→3D反投影(query_view_feats) 总耗时 312.557 ms
[GPU计时] codebook量化 + 3D decoder融合耗时 200.908 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 1.252 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 114.492 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 48.249 ms
[GPU计时] 2D CNN编码(24投影→4尺度2D特征) 耗时 40.744 ms
[GPU计时] 4尺度2D→3D反投影(query_view_feats) 总耗时 321.895 ms
[GPU计时] codebook量化 + 3D decoder融合耗时 245.370 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 1.309 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 160.777 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 3.504 ms
[GPU计时] 2D CNN编码(24投影→4尺度2D特征) 耗时 35.452 ms
[GPU计时] 4尺度2D→3D反投影(query_view_feats) 总耗时 341.244 ms
[GPU计时] codebook量化 + 3D decoder融合耗时 228.391 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.973 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 135.792 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 3.398 ms
[GPU计时] 2D CNN编码(24投影→4尺度2D特征) 耗时 44.281 ms
[GPU计时] 4尺度2D→3D反投影(query_view_feats) 总耗时 297.925 ms
[GPU计时] codebook量化 + 3D decoder融合耗时 202.435 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.942 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 155.275 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 3.464 ms
[GPU计时] 2D CNN编码(24投影→4尺度2D特征) 耗时 36.225 ms
[GPU计时] 4尺度2D→3D反投影(query_view_feats) 总耗时 311.440 ms
[GPU计时] codebook量化 + 3D decoder融合耗时 231.732 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.880 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 164.335 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 3.634 ms
[GPU计时] 2D CNN编码(24投影→4尺度2D特征) 耗时 40.452 ms
[GPU计时] 4尺度2D→3D反投影(query_view_feats) 总耗时 347.763 ms
[GPU计时] codebook量化 + 3D decoder融合耗时 212.973 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.931 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 128.290 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 3.493 ms
[GPU计时] 2D CNN编码(24投影→4尺度2D特征) 耗时 35.614 ms
[GPU计时] 4尺度2D→3D反投影(query_view_feats) 总耗时 408.548 ms
[GPU计时] codebook量化 + 3D decoder融合耗时 280.034 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.937 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 132.115 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 3.487 ms
[GPU计时] 2D CNN编码(24投影→4尺度2D特征) 耗时 35.535 ms
[GPU计时] 4尺度2D→3D反投影(query_view_feats) 总耗时 326.838 ms
[GPU计时] codebook量化 + 3D decoder融合耗时 218.486 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.898 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 153.547 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 3.363 ms
[GPU计时] 2D CNN编码(24投影→4尺度2D特征) 耗时 35.300 ms
[GPU计时] 4尺度2D→3D反投影(query_view_feats) 总耗时 343.217 ms
[GPU计时] codebook量化 + 3D decoder融合耗时 229.428 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.944 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 161.416 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 3.441 ms
[GPU计时] 2D CNN编码(24投影→4尺度2D特征) 耗时 51.335 ms
[GPU计时] 4尺度2D→3D反投影(query_view_feats) 总耗时 336.921 ms
[GPU计时] codebook量化 + 3D decoder融合耗时 215.973 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.899 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 123.225 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 3.495 ms
[GPU计时] 2D CNN编码(24投影→4尺度2D特征) 耗时 41.186 ms
[GPU计时] 4尺度2D→3D反投影(query_view_feats) 总耗时 314.757 ms
[GPU计时] codebook量化 + 3D decoder融合耗时 213.832 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.869 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 127.107 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 3.679 ms
[GPU计时] 2D CNN编码(24投影→4尺度2D特征) 耗时 35.636 ms
[GPU计时] 4尺度2D→3D反投影(query_view_feats) 总耗时 309.975 ms
[GPU计时] codebook量化 + 3D decoder融合耗时 207.262 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.898 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 78.111 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 24.552 ms
```
  
**问题：1w个稀疏点是在训练开始时随机产生的，但是训练过程中（不同epoch），1w个稀疏点的坐标就不再改变了，所以这种也是极为取巧的训练方式**

问题：
- 4尺度2D→3D反投影(query_view_feats) 总耗时 309.975 ms-350 ms 之间，是耗时最多的
- codebook量化 + 3D decoder融合耗时 215 ms 左右，这是第二大耗时最多的
- 考虑到只有稀疏点1w个，且对256^3 有168个1w，所以"query_view_feats×4 2D多视角→稀疏点特征耗时"会×168，大约为140×168=22400ms，这在日常推理中所占时间极大
- 进一步的，"PointDecoder MLP 368→1 HU"所耗费时间也有可能增加

我的下面改进思路
- 首先对于这个1w点的选取，每次要送入不同的、随机的1w个点
- 投影之多2个就可以，也可以就1个低分辨率的体素
  

## 下面是推理时的各路损耗信息
```bash
[GPU计时] 2D CNN编码(24投影→4尺度2D特征) 耗时 3110.521 ms
[GPU计时] 4尺度2D→3D反投影(query_view_feats) 总耗时 38.577 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.721 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 70.949 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 38.787 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.849 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.539 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.458 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.537 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.752 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.452 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.607 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.484 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.539 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.626 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 59.010 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.327 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.527 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.667 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.509 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.609 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.100 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.651 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.620 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.723 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.951 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.671 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.704 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.741 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.613 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.316 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.375 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.616 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.696 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.924 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.753 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.995 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.959 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.655 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.531 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.468 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.565 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.832 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.629 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.593 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.737 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.413 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.580 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.921 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.626 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.652 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.964 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.731 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.595 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.076 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.386 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.637 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.069 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.817 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.681 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.536 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.568 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.640 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.882 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.649 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.676 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.256 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.675 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.604 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.803 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.438 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.640 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.585 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.733 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.652 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.623 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.024 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.595 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.447 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.406 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.578 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.006 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.663 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.631 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.031 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.923 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.523 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.522 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.658 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.640 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.828 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.428 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.537 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.881 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.408 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.572 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.624 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.393 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.604 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.757 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.939 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.636 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 59.551 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.611 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.578 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.217 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.490 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.620 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.811 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.895 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.581 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.663 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.657 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.633 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.567 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.594 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.641 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.055 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.727 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.643 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.772 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.451 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.604 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.621 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.613 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.659 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.146 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.459 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.620 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.128 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.976 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.705 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.211 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.831 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.662 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.494 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.644 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.553 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.821 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.411 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.622 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.500 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.415 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.603 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 56.831 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.931 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.542 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.217 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.143 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.623 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.209 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.310 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.636 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 59.156 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.442 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.591 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.592 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.854 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.613 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.367 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.039 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.569 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.770 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.314 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.818 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 56.437 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.590 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.675 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.767 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.612 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.690 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.597 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.201 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.536 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.843 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.321 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.611 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.858 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.671 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.679 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.714 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.896 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.541 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.551 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.537 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.702 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 55.209 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.384 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.637 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.341 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.689 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.629 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.914 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.120 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.598 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 59.434 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.642 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 1.108 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 59.107 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.519 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.488 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 56.741 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.694 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.618 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 59.365 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.772 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.667 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.165 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.464 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.615 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 59.180 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.428 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.623 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.306 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.825 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.539 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.979 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 18.144 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.587 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.871 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.511 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.588 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.267 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.950 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.649 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.203 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.494 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.910 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 56.949 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.968 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.643 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 59.269 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.432 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.643 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.274 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.433 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.680 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.720 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.713 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.658 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.682 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.364 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.608 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.051 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.558 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.623 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.112 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.887 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.624 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.104 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.740 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.677 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.835 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.465 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.738 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.608 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.364 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.578 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.362 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.646 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.962 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.815 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.841 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.610 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.555 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.462 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.665 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.817 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.745 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.579 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.604 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.515 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.653 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.291 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.069 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.608 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.929 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.340 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.626 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.531 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.535 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.537 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.015 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.590 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.606 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.147 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.762 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.618 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.250 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.786 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.623 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.076 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.386 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.645 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.551 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.541 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.650 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.946 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.558 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.604 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.004 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.578 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.636 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.073 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.893 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.680 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.250 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.464 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.519 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.380 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.516 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.636 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.402 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.587 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.595 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 56.912 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.538 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.556 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.275 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.435 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.580 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 56.808 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.435 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.599 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.715 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.468 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.625 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.094 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 18.028 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.632 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 55.966 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.682 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.606 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.958 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.469 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.550 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.402 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.605 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.599 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 56.381 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.495 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.676 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.774 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.708 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.630 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.738 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.850 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.671 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 56.755 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.443 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.624 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.424 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.430 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.624 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.487 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.801 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.630 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.706 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.921 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 1.125 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.687 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.605 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.649 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.299 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.176 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.575 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.988 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.268 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.643 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.105 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.615 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.553 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 59.652 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.438 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.655 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.734 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.594 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.611 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.022 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.765 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.618 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.329 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.451 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.604 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.782 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.571 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.571 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.807 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.984 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.635 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.266 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.400 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.604 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.906 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.471 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.591 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.559 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.955 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.647 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.688 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.120 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.621 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.505 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 18.137 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.635 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.584 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.406 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.631 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.856 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.554 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.654 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.220 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.822 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.716 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.227 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.723 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.596 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 56.621 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.964 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.597 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.894 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.667 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.672 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.500 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.398 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.611 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.338 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.559 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.904 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 56.530 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.664 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.649 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.239 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.418 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.655 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 56.934 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.329 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.641 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.927 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.760 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.577 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.192 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.395 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.582 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.307 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.862 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.548 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 56.793 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.635 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.748 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.050 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.343 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.538 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.330 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.518 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.637 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.925 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.443 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.530 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.569 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.191 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.637 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.534 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.719 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.652 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.349 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.610 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.635 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.343 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.278 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.902 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.844 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.366 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.573 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.386 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.032 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.926 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.708 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.407 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.627 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.296 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.431 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.538 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.072 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.603 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.568 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.053 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.680 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.639 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 59.024 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.116 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.653 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 57.585 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.932 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.650 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.311 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.866 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.577 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 56.854 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 17.095 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.570 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 58.336 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 16.443 ms
[GPU计时] index_3d 3D体素→稀疏点特征耗时 0.359 ms
[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 43.950 ms
[GPU计时] PointDecoder MLP 368→1 HU预测耗时 13.703 ms
[GPU计时] 完整256³评估总耗时(encode+168批forward_points) 12844 ms = 12.8 s
[GPU计时] 单样本完整推理耗时 16146 ms = 16.1 s
```

### 问题：
- query_view_feats×4 2D多视角→稀疏点特征 依旧是耗费时间的第1大户，56-58ms
- 其次是PointDecoder MLP 368→1 HU 为耗时第2大户 16-17ms
- 可以看到针对1w个点，计算总和才不到80 ms
- 但是如果是168个点完整推理下来，需要×168，看到需要16.1s。这里采用的gpu是4060 8GB 版本。论文采用的是3090 只用了3.1s
- 还有其它的一些中间步骤需要花费时间，难以统计

### 解决 
- 获得1w稀疏点特征这个阶段，如果×168（即得到256^3 个点），则会花费大约1s。如果直接获得128^3 个点,则只会花费1/8s，大约0.22s
- 然后我们再对这128^3 个点进行MLP，可大约为16×168÷(2^3 = 8 ) = 336ms
- 通过上网查资料，得到在 128^3 上做3D CNN Unet，耗时也就 60ms。
- 综上加起来，也就600ms即可完成一次完整的512^3体数据生成，3s内可以做到5次图像生成
- 如果以5090甚至更高的显卡，可以做到稀疏6个视角下，500ms生成一次体数据
- **希望能够在减少维度的同时，加入其它信息来弥补确实的维度信息，比如角度信息**