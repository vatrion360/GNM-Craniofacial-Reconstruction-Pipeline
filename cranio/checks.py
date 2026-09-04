# -*- coding: utf-8 -*-
"""Verificari de consistenta a plasarii markerilor (Etapa 0).

Ruleaza INAINTE de orice fit, ca sa prinda erorile tipice de plasare
(ex. Orbita_Int la centrul orbitei, Pogonion confundat cu Gnathion).
"""

import numpy as np

from .landmarks import CONSISTENCY_PAIRS, side_of


def check_side_swap(targets, rot, trans, scale):
    """Detecteaza markeri Dr/St inversati.

    Dupa aliniere, in spatiul modelului GNM (+X = stanga subiectului),
    un marker '_Dr' trebuie sa aiba x < 0 si unul '_St' x > 0.
    """
    swapped = []
    for label, _, xyz, _ in targets:
        side = side_of(label)
        if side == 0:
            continue
        p_model = (xyz - trans) @ (scale * rot) / (scale ** 2)
        if side == -1 and p_model[0] > 5.0:
            swapped.append(label)
        elif side == 1 and p_model[0] < -5.0:
            swapped.append(label)
    return swapped


def flag_outliers(labels, residuals, min_mm=20.0, factor=2.5):
    """Marcheaza landmarkurile cu reziduu suspicios de mare dupa fit."""
    med = float(np.median(residuals))
    thresh = max(min_mm, factor * med)
    return [(l, float(r)) for l, r in zip(labels, residuals) if r > thresh], thresh


def check_landmark_consistency(targets, mu, label_to_vertex, tol=0.20):
    """Detecteaza markerii plasati gresit din geometria landmarkurilor insesi.

    Pentru fiecare pereche din CONSISTENCY_PAIRS se compara distanta din CSV
    cu cea din template-ul GNM. Scara CSV/template se estimeaza robust ca
    MEDIANA rapoartelor (markerii corecti domina), apoi perechile al caror
    raport normalizat deviaza cu mai mult de ``tol`` sunt suspecte. Metoda
    e independenta de aliniere/rotatie si prinde erorile tipice de plasare
    (ex. Orbita_Int la centrul orbitei -> distanta Dr-St de 2x prea mare).

    Returneaza (rows, suspect) unde rows = lista de (a, b, d_csv,
    d_gnm_scalat, deviatie, este_suspect) pentru raport, iar suspect =
    multimea etichetelor implicate in cel putin o pereche suspecta.
    """
    pos = {label: xyz for label, _, xyz, _ in targets}
    pairs = [(a, b) for a, b in CONSISTENCY_PAIRS if a in pos and b in pos]
    if len(pairs) < 4:
        return [], set()
    d_csv = np.array([np.linalg.norm(pos[a] - pos[b]) for a, b in pairs])
    d_gnm = np.array([np.linalg.norm(mu[label_to_vertex[a]]
                                     - mu[label_to_vertex[b]])
                      for a, b in pairs])
    ratios = d_csv / np.maximum(d_gnm, 1e-9)
    scale_est = float(np.median(ratios))
    rows, suspect = [], set()
    for (a, b), dc, dg in zip(pairs, d_csv, d_gnm * scale_est):
        dev = dc / max(dg, 1e-9) - 1.0
        flag = abs(dev) > tol
        if flag:
            suspect.add(a)
            suspect.add(b)
        rows.append((a, b, float(dc), float(dg), float(dev), flag))
    return rows, suspect
