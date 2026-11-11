"""Identity-free pad normal forces; no interaction with scoring contacts."""


def pad_normal_forces(net_force_w, pad_normal_w):
    """[N,2,3] unfiltered force → [N,2] compression in N; normals point inward."""
    return (-(net_force_w * pad_normal_w).sum(-1)).clip(0, None)
