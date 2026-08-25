# -*- coding: utf-8 -*-
"""Raportul TXT de reconstructie (format identic cu v3.1)."""

import datetime
import os

import numpy as np


def write_stats(path, cfg, targets, skipped, scale, lam_used, fit_info,
                res_align, res_fit, res_final, clamped_mags, field, c,
                warnings_list, extra, dense_report=None, lm_regions=None,
                consistency=None, excluded_auto=None):
    """Raport complet de reconstructie (reproductibilitate / publicatie).

    ``cfg`` este un PipelineConfig (are aceleasi nume de attribute ca
    vechiul obiect ``args`` din v3.1).
    """
    loo_table, history, dense_stats = fit_info
    lines = []
    lines.append("=== Statistici Reconstructie Craniofaciala GNM v3.1 ===")
    lines.append(f"Generat: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"CSV intrare: {os.path.abspath(cfg.input)}")
    lines.append(f"Model: {os.path.abspath(cfg.npz)}")
    lines.append(f"Mesh OBJ: {os.path.abspath(cfg.output)}")
    lines.append("")
    lines.append(f"Markeri folositi: {len(targets)}")
    if skipped:
        lines.append("Markeri exclusi:")
        for label, reason in skipped:
            lines.append(f"  - {label}: {reason}")
    if excluded_auto:
        lines.append("Markeri exclusi automat (--exclude-outliers):")
        for label, res in excluded_auto:
            lines.append(f"  - {label}: reziduu la primul fit {res:.1f} mm")
    if consistency:
        lines.append("")
        lines.append("Verificare consistenta plasare (distante CSV vs "
                     "template GNM scalat la mediana rapoartelor):")
        lines.append(f"{'Pereche':48s} {'CSV':>8s} {'GNM*':>8s} {'Dev':>7s}")
        for a, b, dc, dg, dev, flag in consistency:
            mark = "  SUSPECT" if flag else ""
            lines.append(f"{(a + ' - ' + b):48s} {dc:8.1f} {dg:8.1f} "
                         f"{dev * 100:+6.0f}%{mark}")
    lines.append("")
    lines.append(f"Etapa 1 - Aliniere: scala = {scale:.4f} "
                 f"(RMS = {res_align.mean():.2f} mm)")
    lines.append(f"Etapa 2 - Fit statistic: lambda = {lam_used:g}, "
                 f"|c|max = {np.abs(c).max():.2f} sigma, "
                 f"|c|med = {np.abs(c).mean():.2f} sigma, "
                 f"RMS = {res_fit.mean():.2f} mm")
    if loo_table:
        lines.append("  LOO-CV (eroare medie predictie -> lambda):")
        for err, lam in loo_table:
            marker = "  <-- ales" if lam == lam_used else ""
            lines.append(f"    {err:7.2f} mm  lambda={lam:g}{marker}")
    lines.append(f"  Convergenta fit (iter, scala, RMS mm, |c|max):")
    for it, s, rms, cmax in history:
        lines.append(f"    it={it:2d}  scala={s:.4f}  RMS={rms:6.2f}  |c|max={cmax:.2f}")
    lines.append(f"Etapa 3 - Corectie locala (cap fata = "
                 f"{cfg.face_cap_mm:g} mm, scalp = "
                 f"{cfg.max_correction_mm:g} mm, damping zone protejate = "
                 f"x{cfg.protect_damping:g}): "
                 f"medie = {np.linalg.norm(field, axis=1).mean():.2f} mm, "
                 f"max = {np.linalg.norm(field, axis=1).max():.2f} mm, "
                 f"RMS final la markeri = {res_final.mean():.2f} mm")
    if dense_report:
        lines.append("")
        lines.extend(dense_report)
    lines.append("")
    lines.append(f"{'Landmark':22s} {'Regiune':14s} {'Vertex':>7s} "
                 f"{'Aliniere':>9s} {'Fit':>9s} {'Final':>9s} {'Prescris':>10s}")
    lines.append("-" * 90)
    for (label, vid, _, _), ra, rf, rn, cm in zip(
            targets, res_align, res_fit, res_final, clamped_mags):
        reg = lm_regions.get(label, "-") if lm_regions else "-"
        lines.append(f"{label:22s} {reg:14s} {vid:7d} {ra:8.2f}  {rf:8.2f}  "
                     f"{rn:8.2f}  {cm:9.2f}")
    lines.append("-" * 74)
    if extra:
        lines.append("")
        lines.append(extra)
    if warnings_list:
        lines.append("")
        lines.append("AVERTISMENTE:")
        for w in warnings_list:
            lines.append(f"  [!] {w}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
