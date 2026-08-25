# -*- coding: utf-8 -*-
"""Citire/scriere fisiere de markeri craniofaciali.

Formate suportate:

    * legacy v11/v12 - CSV simplu cu coloanele
      ``gnm_landmark_index,x,y,z`` (index codificat cu semn; sentinelul
      ``0,0,0`` = marker neplasat). Fara metadate.
    * v2 - acelasi CSV, prefixat de doua linii-comentariu::

        # gnm-marker-csv v2
        # {"units": "mm", "addon_version": "12.0.0", ...}

      Metadatele (JSON, o singura linie) includ unitatile, versiunea
      addon-ului, fisierul craniu sursa si adancimile de tesut folosite -
      pentru reproductibilitate (TODO.md, Faza 10). Cititorul legacy
      ramane neschimbat: fisierele vechi se citesc identic.
"""

import csv
import json
from typing import List, NamedTuple, Tuple

import numpy as np

from .landmarks import CONFIDENCE_WEIGHTS, DEFAULT_CONFIDENCE

V2_MAGIC = "# gnm-marker-csv v2"


class MarkerTarget(NamedTuple):
    """Un marker plasat: eticheta, vertexul modelului, tinta, ponderea.

    NamedTuple ca sa ramana compatibil cu decompozitia positionala
    ``(label, vertex, xyz, weight)`` din codul existent.
    """
    label: str
    vertex: int
    xyz: np.ndarray
    weight: float


def _decode_rows(rows, index_to_label, label_to_vertex):
    """Transforma randuri CSV in (targets, skipped) - logica comuna."""
    targets: List[MarkerTarget] = []
    skipped: List[Tuple[str, str]] = []
    for row in rows:
        enc = int(row["gnm_landmark_index"])
        x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
        label = index_to_label.get(enc)
        if label is None:
            skipped.append((str(enc),
                            "index necunoscut (versiune addon mai noua?)"))
            continue
        if abs(x) < 1e-9 and abs(y) < 1e-9 and abs(z) < 1e-9:
            skipped.append((label, "neplasat in addon (0,0,0)"))
            continue
        targets.append(MarkerTarget(
            label=label,
            vertex=label_to_vertex[label],
            xyz=np.array([x, y, z], dtype=np.float64),
            weight=CONFIDENCE_WEIGHTS.get(label, DEFAULT_CONFIDENCE),
        ))
    return targets, skipped


def read_marker_csv(csv_path, index_to_label, label_to_vertex):
    """Citeste un fisier de markeri (legacy sau v2), cu auto-detectie.

    Returneaza (targets, skipped, metadata) unde:
      targets = lista de MarkerTarget;
      skipped = lista de (eticheta, motiv) pentru markerii exclusi;
      metadata = dict ({"version": 1} pentru fisiere legacy).
    """
    with open(csv_path, newline="") as f:
        raw_lines = f.read().splitlines()

    metadata = {"version": 1}
    body = raw_lines
    if raw_lines and raw_lines[0].strip() == V2_MAGIC:
        metadata = {"version": 2}
        if len(raw_lines) > 1 and raw_lines[1].lstrip().startswith("#"):
            try:
                metadata.update(json.loads(raw_lines[1].lstrip()[1:].strip()))
            except json.JSONDecodeError:
                metadata["metadata_error"] = "linia JSON de metadate e corupta"
        body = [ln for ln in raw_lines if not ln.lstrip().startswith("#")]

    targets, skipped = _decode_rows(
        csv.DictReader(body), index_to_label, label_to_vertex)
    return targets, skipped, metadata


def write_marker_csv_v2(csv_path, rows, metadata=None):
    """Scrie un fisier de markeri v2 (metadate JSON + CSV legacy-compatibil).

    rows = lista de (index_codificat, x, y, z); sunt sortate ca la exportul
    legacy. Corpul CSV ramane citibil de orice cititor legacy care ignora
    liniile comentariu.
    """
    meta = {"version": 2}
    if metadata:
        meta.update(metadata)
    with open(csv_path, "w", newline="") as f:
        f.write(V2_MAGIC + "\n")
        f.write("# " + json.dumps(meta, ensure_ascii=False, sort_keys=True)
                + "\n")
        writer = csv.writer(f)
        writer.writerow(["gnm_landmark_index", "x", "y", "z"])
        writer.writerows(sorted(rows))
    return meta


def load_csv_targets(csv_path, index_to_label, label_to_vertex):
    """Compatibilitate cu vechiul API: returneaza (targets, skipped).

    targets este o lista de tuple-uri (eticheta, vertex, xyz, greutate),
    exact ca in gnm_reconstruct v3.1.
    """
    targets, skipped, _meta = read_marker_csv(
        csv_path, index_to_label, label_to_vertex)
    tuples = [(t.label, t.vertex, t.xyz, t.weight) for t in targets]
    return tuples, skipped
