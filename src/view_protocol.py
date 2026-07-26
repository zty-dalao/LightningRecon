"""Shared deterministic view-selection protocol."""

VIEW_CURRICULA = {
    6: (60, 54, 48, 36, 24, 12, 6),
    8: (64, 56, 48, 32, 24, 16, 8),
    10: (60, 50, 40, 30, 20, 10),
}


def uniform_view_indices(total_views, n_views):
    """Return fixed full-rotation indices using floor(k * total / n)."""
    if total_views <= 0:
        raise ValueError(f'total_views must be positive, got {total_views}')
    if n_views <= 0 or n_views > total_views:
        raise ValueError(
            f'n_views must be in [1, {total_views}], got {n_views}'
        )
    return tuple(k * total_views // n_views for k in range(n_views))


def resolve_view_curriculum(final_view, override=None, max_views=64):
    """Resolve and validate a high-to-low, target-anchored view curriculum."""
    if final_view not in VIEW_CURRICULA:
        supported = ', '.join(str(v) for v in sorted(VIEW_CURRICULA))
        raise ValueError(
            f'Unsupported final_view={final_view}; supported values: {supported}'
        )

    if override:
        if isinstance(override, str):
            try:
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
            views = tuple(int(value) for value in override)
    else:
        views = VIEW_CURRICULA[final_view]

    if len(views) < 2:
        raise ValueError('view_schedule must contain at least two view counts')
    if views[0] > max_views:
        raise ValueError(
            f'view_schedule starts at {views[0]}, above max_views={max_views}'
        )
    if views[-1] != final_view:
        raise ValueError(
            f'view_schedule must end at final_view={final_view}, got {views[-1]}'
        )
    if any(high <= low for high, low in zip(views, views[1:])):
        raise ValueError(
            f'view_schedule must be strictly descending, got {list(views)}'
        )
    incompatible = [views_count for views_count in views if views_count % final_view]
    if incompatible:
        raise ValueError(
            'Every view count must be a multiple of final_view so the deployment '
            f'angles remain present; incompatible values: {incompatible}'
        )
    return views
