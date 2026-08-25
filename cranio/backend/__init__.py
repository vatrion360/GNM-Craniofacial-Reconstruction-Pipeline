# -*- coding: utf-8 -*-
"""Backend-uri de model facial statistic.

Arhitectura este independenta de un model concret (TODO.md, Faza 6):
``base.FaceModelBackend`` defineste interfata; ``gnm_backend.GNMBackend``
este prima implementare (GNM Head v3.0, Google).
"""

from .base import FaceModelData, FaceModelBackend
from .gnm_backend import GNMBackend, default_npz_path

__all__ = [
    "FaceModelData",
    "FaceModelBackend",
    "GNMBackend",
    "default_npz_path",
]
