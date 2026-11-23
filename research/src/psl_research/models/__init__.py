"""Model zoo. Each module registers its model(s) in MODELS; import it here.

Every model takes (batch, time, features) and returns (batch, num_classes) logits,
and is constructed as Model(input_size=..., num_classes=..., **params).
"""

from . import tcn  # noqa: F401
