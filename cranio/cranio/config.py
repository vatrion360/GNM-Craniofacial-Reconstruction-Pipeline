# -*- coding: utf-8 -*-
"""Configuratia completa a unei rulari de reconstructie (PipelineConfig).

Inlocuieste imprastierea de argumente argparse din v3.1: un singur obiect
de date, cu aceleasi nume de attribute ca vechiul ``args``, care traverseaza
pipeline-ul si ajunge intact in raport (reproductibilitate).
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

from .backend import default_npz_path


@dataclass
class PipelineConfig:
    """Toate optiunile pipeline-ului, cu implicitele din v3.1."""

    # Intrari/iesiri
    input: str = ""
    output: Optional[str] = None
    output_error_mesh: Optional[str] = None
    output_stats: Optional[str] = None
    npz: str = field(default_factory=default_npz_path)
    skull: Optional[str] = None

    # Fit statistic
    regularization: str = "auto"          # "auto" (LOO-CV) sau valoare fixa
    exclude: List[str] = field(default_factory=list)
    exclude_outliers: bool = False

    # Corectie locala TPS
    skip_tps: bool = False
    max_correction_mm: float = 15.0       # cap neted per-vertex pe scalp
    face_cap_mm: float = 8.0              # cap neted per-vertex pe fata
    protect_damping: float = 0.25         # amortizare zone fara ancore

    # Constrangeri dense (necesita skull)
    scalp_offset_mm: float = 5.0
    dense_weight: float = 0.5
    dense_samples: int = 200000
    tps_scalp_centres: int = 500
    tps_face_centres: int = 200
    no_face_dense: bool = False
    no_dense_fit: bool = False
    no_dense_tps: bool = False

    # Termeni optionali de loss (0.0 = dezactivat, comportament v3.1)
    symmetry_weight: float = 0.0
    distance_weight: float = 0.0
    prior_soft_sigma: float = 0.0
    prior_soft_weight: float = 4.0

    def fill_default_outputs(self):
        """Completeaza caile de iesire implicite din numele intrarii."""
        if self.output is None:
            base, _ = os.path.splitext(self.input)
            self.output = base + "_reconstructie.obj"
        if self.output_error_mesh is None:
            base, _ = os.path.splitext(self.output)
            self.output_error_mesh = base + "_heatmap.ply"
        if self.output_stats is None:
            base, _ = os.path.splitext(self.output)
            self.output_stats = base + "_statistici.txt"
