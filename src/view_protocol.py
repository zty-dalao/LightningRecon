"""视角课程与确定性评估协议。

本文件只负责“选择哪些视角”，不读取投影，也不改变投影与物理角度的配对关系。
训练和推理共同调用这里的函数，避免二者分别实现采样后产生协议偏差。
"""

# 内置课程按从高视角到最终部署视角排列。
# 第一个数字是均匀基准集大小，最后一个数字必须等于 final_view。
VIEW_CURRICULA = {
    6: (60, 54, 48, 36, 24, 12, 6),
    8: (64, 56, 48, 32, 24, 16, 8),
    10: (60, 50, 40, 30, 20, 10),
}


def uniform_view_indices(total_views, n_views):
    """在完整源视角网格上确定性地选取近似等间隔索引。

    第 ``k`` 个索引使用 ``floor(k * total_views / n_views)``。该公式不会
    选择越界索引，并且相同输入始终得到相同结果，适合验证、测试和部署。
    """
    # 源投影数量必须为正数，否则不存在可采样的视角网格。
    if total_views <= 0:
        raise ValueError(f'total_views must be positive, got {total_views}')
    # 既不允许空视角，也不通过填充制造并不存在的投影。
    if n_views <= 0 or n_views > total_views:
        raise ValueError(
            f'n_views must be in [1, {total_views}], got {n_views}'
        )
    # 使用整数除法等价于 floor；返回 tuple 便于写入 checkpoint 元数据。
    return tuple(k * total_views // n_views for k in range(n_views))


def resolve_view_curriculum(final_view, override=None, max_views=64):
    """解析并校验从高视角到 ``final_view`` 的训练课程。

    ``override`` 为空时使用内置映射；非空时接受逗号分隔字符串或整数序列。
    返回值始终是严格递减的 tuple，训练脚本据此创建各阶段 epoch 区间。
    """
    # 只接受已经定义了厂家部署协议的最终视角数。
    if final_view not in VIEW_CURRICULA:
        supported = ', '.join(str(v) for v in sorted(VIEW_CURRICULA))
        raise ValueError(
            f'Unsupported final_view={final_view}; supported values: {supported}'
        )

    # 命令行覆盖优先于内置课程，但仍必须通过下面的全部约束。
    if override:
        if isinstance(override, str):
            try:
                # 忽略逗号两侧空白和空字段，例如 "60, 48,24,12,6"。
                views = tuple(
                    int(value.strip())
                    for value in override.split(',')
                    if value.strip()
                )
            except ValueError as exc:
                raise ValueError(
                    f'Invalid view_schedule={override!r}; expected comma-separated integers'
                ) from exc
        else:
            # 测试或 Python API 可直接传入 list/tuple。
            views = tuple(int(value) for value in override)
    else:
        views = VIEW_CURRICULA[final_view]

    # 至少包含一个高视角阶段和最终稀疏视角阶段。
    if len(views) < 2:
        raise ValueError('view_schedule must contain at least two view counts')
    # 4090D 的当前设计上限为 64 views。
    if views[0] > max_views:
        raise ValueError(
            f'view_schedule starts at {views[0]}, above max_views={max_views}'
        )
    # 课程的最后一个视角数必须与训练、验证和部署目标一致。
    if views[-1] != final_view:
        raise ValueError(
            f'view_schedule must end at final_view={final_view}, got {views[-1]}'
        )
    # 禁止重复或回升的视角数，确保课程始终由高到低。
    if any(high <= low for high, low in zip(views, views[1:])):
        raise ValueError(
            f'view_schedule must be strictly descending, got {list(views)}'
        )
    # 基准视角数能整除 final_view 时，固定部署子集可严格嵌套于基准集。
    if views[0] % final_view:
        raise ValueError(
            'The maximum-view base must be divisible by final_view so the '
            f'fixed deployment subset is well-defined; got {views[0]} and '
            f'{final_view}'
        )
    # 后续代码只读此不可变序列，不在训练过程中修改课程。
    return views
