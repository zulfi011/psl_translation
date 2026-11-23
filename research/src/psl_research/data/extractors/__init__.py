"""Landmark extractors: raw video → (T, F) array saved under data/interim/<extractor name>/."""

from .base import Extractor  # noqa: F401
from . import mediapipe_holistic  # noqa: F401  (registers "mediapipe_holistic_v1")
