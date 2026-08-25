#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNM Reconstructie Craniofaciala (v4.0)
======================================

Pipeline stiintific de reconstructie faciala (facial approximation) pornind de
la markerii craniofaciali plasati pe un craniu scanat cu addon-ul Blender
``addon_v11.py`` / ``addon_v12.py`` si de la modelul statistic de cap uman
GNM Head v3.0 (Google).

Noutati v4.0 - refactorizare in pachetul ``cranio`` + termeni noi de loss:
    * Toata logica numerica a fost extrasa neschimbata in pachetul pur-Python
      ``cranio`` (fara dependinte de Blender, testabil): registru anatomic
      unic (cranio.landmarks), backend de model abstract
      (cranio.backend.GNMBackend), optimizor cu termeni de loss conectabili
      (cranio.optimize), verificari, geometrie, exporturi, raport.
      Acest script este doar un CLI subtire peste cranio.pipeline.
    * Termeni optionali de loss (implicit dezactivati = rezultate identice
      cu v3.1): --symmetry-weight (prior de simetrie bilaterala in spatiul
      latent), --distance-weight (constrangeri de distanta intre perechile
      de landmarkuri, tinta = template-ul statistic),
      --prior-soft-sigma (prior latent moale peste prag, in locul singurului
      clip dur la 3 sigma).
    * Diagnostice de stabilitate: avertismente explicite cand LOO-CV alege
      lambda la marginea grilei sau cand coeficientii ating clipul dur.
    * Cititor CSV v2 (metadate JSON: unitati, versiune addon, fisier craniu,
      adancimi de tesut) cu compatibilitate legacy v11/v12.

Noutati v3.1 - robustete la markeri plasati gresit + TPS region-aware:
    * Verificare de consistenta a plasarii (Etapa 0): rapoartele distantelor
      inter-landmark CSV vs template GNM (normalizate robust la scara) scot
      in evidenta markerii plasati gresit INAINTE de fit.
    * ``--exclude LABEL...`` si ``--exclude-outliers``: markerii cu reziduu
      mare pot fi exclusi complet din fit si din centrele TPS.
    * Corectia TPS tine cont de CONFIDENCE_WEIGHTS, are cap per-vertex si
      amortizare pe geometria fara ancore anatomice (ochi/buze/gura).

Noutati v3.0 / v2.0 - constrangeri dense de suprafata (optional, --skull):
    atractia de craniu pe scalp + regiuni faciale cu tesut SUBTIRE, fiecare
    cu offsetul ei (brow ~5mm, punte nazala ~3mm, zigomatic ~8.5mm,
    barbie ~10mm, infraorbital ~7mm), cu respingere automata a
    corespondentelor invalide.

Metodologie (in 4 etape):
    0.  Incarcare si verificari (consistenta plasarii markerilor).
    1.  Aliniere de similaritate (Umeyama 1991, ponderata, robusta).
    2.  Fit statistic: estimare alternativa a coeficientilor de identitate
        GNM (ridge LSQ ponderat, IRLS) si a transformarii de similaritate;
        lambda ales prin LOO-CV daca --regularization auto.
    3.  Corectie locala limitata (hibrid): camp TPS (Bookstein 1989) cu o
        singura limitare neta (tanh) si cap per-vertex.
    4.  Export OBJ + PLY heatmap + TXT cu statistici complete.

Dependente: numpy, scipy, trimesh (doar pentru incarcarea craniului).
Rulare din consola (CPython), NU din Blender.
"""

import argparse
import sys

from cranio.backend import default_npz_path
from cranio.config import PipelineConfig
from cranio.pipeline import run_pipeline


def parse_args(argv=None) -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Reconstructie faciala GNM din markeri craniofaciali (CSV addon Blender).")
    parser.add_argument("--input", required=True, help="CSV exportat de addon (v11/v12/v2)")
    parser.add_argument("--output", default=None,
                        help="OBJ de iesire (implicit: <input>_reconstructie.obj)")
    parser.add_argument("--output-error-mesh", default=None,
                        help="PLY heatmap al corectiei locale (implicit: <output>_heatmap.ply)")
    parser.add_argument("--output-stats", default=None,
                        help="TXT cu statistici (implicit: <output>_statistici.txt)")
    parser.add_argument("--npz", default=default_npz_path(),
                        help="Calea catre gnm_head.npz")
    parser.add_argument("--regularization", default="auto",
                        help="'auto' (LOO-CV) sau o valoare fixa, ex. 30")
    parser.add_argument("--exclude", nargs="+", default=[], metavar="LABEL",
                        help="Etichete de markeri exclusi manual din fit si din "
                             "centrele TPS (ex. --exclude Pogonion Rhinion)")
    parser.add_argument("--exclude-outliers", action="store_true",
                        help="Exclude automat markerii cu reziduu > max(15 mm, "
                             "3*MAD) dupa primul fit si reface fitul o data; "
                             "implicit sunt doar semnalati, nu exclusi")
    parser.add_argument("--max-correction-mm", type=float, default=15.0,
                        help="Limita neteda a corectiei locale per vertex pe "
                             "SCALP (mm); pe fata se foloseste --face-cap-mm")
    parser.add_argument("--face-cap-mm", type=float, default=8.0,
                        help="Limita neteda a corectiei locale per vertex pe "
                             "FATA (ochi/nas/gura), in mm")
    parser.add_argument("--protect-damping", type=float, default=0.25,
                        help="Factor de amortizare a corectiei TPS pe zonele "
                             "fara ancore anatomice (ochi/interior gura/buze); "
                             "1.0 = fara protectie")
    parser.add_argument("--skip-tps", action="store_true",
                        help="Opreste dupa fitul statistic (fara corectie locala)")
    parser.add_argument("--skull", default=None,
                        help="Craniu (STL/OBJ) din aceeasi scena Blender ca CSV-ul "
                             "(world, mm) - activeaza constrangerile dense de scalp")
    parser.add_argument("--scalp-offset-mm", type=float, default=5.0,
                        help="Offset tesut moale piele-os pe scalpa (mm)")
    parser.add_argument("--dense-weight", type=float, default=0.5,
                        help="Ponderea totala a constrangerilor dense, relativa la "
                             "suma ponderilor markerilor")
    parser.add_argument("--dense-samples", type=int, default=200000,
                        help="Nr. de puncte esantionate pe suprafata craniului")
    parser.add_argument("--tps-scalp-centres", type=int, default=500,
                        help="Nr. maxim de puncte de scalp folosite ca centre TPS")
    parser.add_argument("--tps-face-centres", type=int, default=200,
                        help="Nr. maxim de puncte faciale folosite ca centre TPS")
    parser.add_argument("--no-face-dense", action="store_true",
                        help="Constrangeri dense doar pe scalp (ca v2), nu si pe "
                             "regiunile faciale cu tesut subtire")
    parser.add_argument("--no-dense-fit", action="store_true",
                        help="Fara constrangeri dense in fitul statistic")
    parser.add_argument("--no-dense-tps", action="store_true",
                        help="Fara centre dense in corectia TPS")
    parser.add_argument("--symmetry-weight", type=float, default=0.0,
                        help="Ponderea priorului de simetrie bilaterala in "
                             "spatiul latent (relativa la lambda); 0 = inactiv")
    parser.add_argument("--distance-weight", type=float, default=0.0,
                        help="Ponderea totala a constrangerilor de distanta "
                             "inter-landmark (tinta = template-ul), relativa la "
                             "suma ponderilor markerilor; 0 = inactiv")
    parser.add_argument("--prior-soft-sigma", type=float, default=0.0,
                        help="Prag (sigma) peste care se activeaza priorul "
                             "latent moale; 0 = inactiv (ramane doar clipul dur)")
    parser.add_argument("--prior-soft-weight", type=float, default=4.0,
                        help="Intensitatea priorului moale (multiplu de lambda)")
    args = parser.parse_args(argv)

    return PipelineConfig(
        input=args.input,
        output=args.output,
        output_error_mesh=args.output_error_mesh,
        output_stats=args.output_stats,
        npz=args.npz,
        skull=args.skull,
        regularization=args.regularization,
        exclude=args.exclude,
        exclude_outliers=args.exclude_outliers,
        skip_tps=args.skip_tps,
        max_correction_mm=args.max_correction_mm,
        face_cap_mm=args.face_cap_mm,
        protect_damping=args.protect_damping,
        scalp_offset_mm=args.scalp_offset_mm,
        dense_weight=args.dense_weight,
        dense_samples=args.dense_samples,
        tps_scalp_centres=args.tps_scalp_centres,
        tps_face_centres=args.tps_face_centres,
        no_face_dense=args.no_face_dense,
        no_dense_fit=args.no_dense_fit,
        no_dense_tps=args.no_dense_tps,
        symmetry_weight=args.symmetry_weight,
        distance_weight=args.distance_weight,
        prior_soft_sigma=args.prior_soft_sigma,
        prior_soft_weight=args.prior_soft_weight,
    )


def main(argv=None) -> int:
    return run_pipeline(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
