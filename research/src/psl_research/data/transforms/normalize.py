import numpy as np

from ...registry import TRANSFORMS
from ..layout import L_SHOULDER, R_SHOULDER, join_blocks, present, split_blocks


@TRANSFORMS.register("normalize_body")
class NormalizeBody:
    """
    Make coordinates independent of where the signer stands and how far from the
    camera: centre on the shoulder midpoint, divide by shoulder width (per clip,
    median over frames). Undetected blocks stay exactly zero.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, x, rng):
        left, right, pose = (b.copy() for b in split_blocks(x))
        pmask = present(pose)
        if not pmask.any():
            return x
        ls, rs = pose[pmask, L_SHOULDER, :2], pose[pmask, R_SHOULDER, :2]
        center = np.median((ls + rs) / 2, axis=0)
        scale = max(np.median(np.linalg.norm(ls - rs, axis=1)), self.eps)
        for block, ncoord in ((left, 3), (right, 3), (pose, 3)):
            m = present(block)
            block[m, :, :2] = (block[m, :, :2] - center) / scale
            block[m, :, 2] = block[m, :, 2] / scale
        return join_blocks(left, right, pose)


@TRANSFORMS.register("presence_flags")
class PresenceFlags:
    """Append 3 binary features (left hand, right hand, pose detected) so the model
    can tell 'not detected' apart from a real coordinate of 0."""

    def __call__(self, x, rng):
        left, right, pose = split_blocks(x)
        flags = np.stack([present(left), present(right), present(pose)], axis=1).astype(np.float32)
        return np.concatenate([x, flags], axis=1)
