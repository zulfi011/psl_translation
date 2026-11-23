import numpy as np

from psl_research.data.layout import split_blocks
from psl_research.data.transforms import Pipeline
from psl_research.registry import TRANSFORMS

RNG = np.random.default_rng(0)


def clip(T=40, left=True, right=True):
    x = RNG.uniform(0.2, 0.8, (T, 258)).astype(np.float32)
    x[:, 65:126:3] = -0.05            # negative z on right hand
    if not left:
        x[:, 0:63] = 0
    if not right:
        x[:, 63:126] = 0
    return x


def test_mirror_keeps_missing_hand_zero_and_swaps_hands():
    x = clip(left=False)
    y = TRANSFORMS.build("mirror", p=1.0)(x, RNG)
    left, right, _ = split_blocks(y)
    assert np.all(right == 0)                                  # missing left hand → now the (still empty) right slot
    assert np.allclose(left[..., 0], 1 - split_blocks(x)[1][..., 0])


def test_mirror_twice_is_identity():
    x = clip()
    m = TRANSFORMS.build("mirror", p=1.0)
    assert np.allclose(m(m(x, RNG), RNG), x)


def test_noise_does_not_touch_missing_or_clip_z():
    x = clip(left=False)
    y = TRANSFORMS.build("gaussian_noise", p=1.0, sigma=0.01)(x, RNG)
    assert np.all(y[:, 0:63] == 0)
    assert (y[:, 65:126:3] < 0).all()


def test_fill_missing_hands_forward_fills():
    x = clip()
    x[10:15, 63:126] = 0
    y = TRANSFORMS.build("fill_missing_hands")(x, RNG)
    assert np.allclose(y[10:15, 63:126], x[9, 63:126])


def test_normalize_body_centres_shoulders():
    y = TRANSFORMS.build("normalize_body")(clip(), RNG)
    pose = split_blocks(y)[2]
    mid = (pose[:, 11, :2] + pose[:, 12, :2]) / 2
    assert abs(np.median(mid)) < 1e-4


def test_pipeline_output_shape_and_train_only():
    spec = [{"name": "mirror", "train_only": True, "params": {"p": 1.0}},
            {"name": "presence_flags"}, {"name": "resample", "params": {"length": 60}}]
    x = clip(T=97)
    assert Pipeline(spec, train=False)(x).shape == (60, 261)
    assert not np.allclose(Pipeline(spec, train=True)(x), Pipeline(spec, train=False)(x))
