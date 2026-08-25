# -*- coding: utf-8 -*-
"""Fluxul complet de reconstructie end-to-end (Etapele 0-4).

Asambleaza modulele pachetului cranio intr-un singur apel:
    run_pipeline(PipelineConfig) -> cod de iesire (0 = succes, 2 = fatal).

Folosit atat de CLI (gnm_reconstruct.py), cat - in perspectiva M5 - de
addon-ul Blender. Nu depinde de argparse si nu stie nimic de bpy.
"""

import os

import numpy as np

from .backend import GNMBackend
from .checks import (check_landmark_consistency, check_side_swap,
                     flag_outliers)
from .config import PipelineConfig
from .export import export_heatmap_ply, export_obj
from .geometry import (build_face_dense_regions, build_protected_mask,
                       build_scalp_mask, build_vertex_region_map,
                       dense_correspondences, load_skull_samples)
from .io_csv import read_marker_csv
from .landmarks import (CONSISTENCY_PAIRS, PLACEMENT_HINTS,
                        REGION_OFFSETS_MM)
from .optimize import (LossConfig, bounded_tps_correction, fit_identity,
                       robust_alignment, stability_warnings)
from .report import write_stats


def run_pipeline(cfg: PipelineConfig) -> int:
    """Ruleaza pipeline-ul complet. Returneaza 0 (succes) sau 2 (fatal)."""
    cfg.fill_default_outputs()
    warnings_list = []

    backend = GNMBackend(cfg.npz)
    label_to_vertex = backend.landmark_vertex_map

    loss_cfg = LossConfig(
        symmetry_weight=cfg.symmetry_weight,
        distance_weight=cfg.distance_weight,
        prior_soft_sigma=cfg.prior_soft_sigma,
        prior_soft_weight=cfg.prior_soft_weight,
    )

    # --- Etapa 0: incarcare si verificari ---------------------------------
    print(f"[0] Incarc CSV: {cfg.input}")
    targets, skipped, _csv_meta = read_marker_csv(
        cfg.input, backend.index_to_label, label_to_vertex)
    if cfg.exclude:
        excl = set(cfg.exclude)
        skipped.extend((t[0], "exclus manual (--exclude)")
                       for t in targets if t[0] in excl)
        targets = [t for t in targets if t[0] not in excl]
        unknown = excl - {t[0] for t in targets} - {l for l, _ in skipped}
        if unknown:
            warnings_list.append(
                "--exclude: etichete negasite in CSV: " + ", ".join(sorted(unknown)))
    for label, reason in skipped:
        print(f"    - exclus {label}: {reason}")
    if len(targets) < 4:
        print(f"[EROARE FATALA] Prea putini markeri valizi ({len(targets)}). "
              f"Sunt necesari minim 4 (recomandat >=10).")
        return 2
    if len(targets) < 10:
        warnings_list.append(
            f"Doar {len(targets)} markeri valizi (<10) - reconstructia va fi "
            f"slab constransa si dominata de media statistica.")

    print(f"[0] Incarc modelul GNM: {cfg.npz}")
    model = backend.load()
    mu, basis = model.mu, model.basis
    triangles = model.triangles
    vertex_groups, vertex_group_names = (model.vertex_groups,
                                         model.vertex_group_names)
    labels = [t[0] for t in targets]
    lm_idx = np.array([t[1] for t in targets], dtype=np.int64)
    targets_xyz = np.array([t[2] for t in targets], dtype=np.float64)
    weights = np.array([t[3] for t in targets], dtype=np.float64)

    # Verificare de consistenta a plasarii (distante inter-landmark vs
    # template): markerii suspecti sunt semnalati INAINTE de orice fit.
    cons_rows, cons_suspect = check_landmark_consistency(
        targets, mu, label_to_vertex)
    if cons_rows:
        n_flag = sum(1 for r in cons_rows if r[5])
        print(f"[0] Verificare consistenta plasare: {len(cons_rows)} perechi, "
              f"{n_flag} suspecte")
        for a, b, dc, dg, dev, flag in cons_rows:
            if flag:
                print(f"    [!] {a} - {b}: {dc:.1f} mm vs {dg:.1f} mm "
                      f"asteptat ({dev:+.0%})")
        for label in sorted(cons_suspect):
            hint = PLACEMENT_HINTS.get(label)
            msg = f"Marker suspect de plasare gresita: {label}"
            if hint:
                msg += f". Pozitia corecta: {hint}"
            msg += " (verifica in Blender; poti folosi --exclude sau --exclude-outliers)"
            warnings_list.append(msg)

    # --- Craniu (optional): constrangeri dense de suprafata ---------------
    dense = None
    if cfg.skull:
        if not os.path.exists(cfg.skull):
            print(f"[EROARE FATALA] Fisierul craniu nu exista: {cfg.skull}")
            return 2
        print(f"[0] Incarc craniul: {cfg.skull}")
        from scipy.spatial import cKDTree
        skull_mesh, sk_points, sk_normals = load_skull_samples(
            cfg.skull, cfg.dense_samples)
        tree = cKDTree(sk_points)
        scalp_idx = build_scalp_mask(mu, vertex_groups, vertex_group_names)
        face_regions = ([] if cfg.no_face_dense else
                        build_face_dense_regions(mu, vertex_groups,
                                                 vertex_group_names))
        # Regiunile faciale PRIMELE (offseturi mai specifice); un vertex
        # prezent in mai multe regiuni (ex. scalp ∩ zigomatic) apare O SINGURA
        # DATA - altfel ar deveni doua centre TPS identice cu tinte diferite
        # (matrice aproape singulara -> camp oscilant).
        region_names = [r[0] for r in face_regions] + ["scalp"]
        region_idx = [r[1] for r in face_regions] + [scalp_idx]
        region_offs = [np.full(len(r[1]), r[2]) for r in face_regions]
        region_offs += [np.full(len(scalp_idx), cfg.scalp_offset_mm)]
        dense_idx_all = np.concatenate(region_idx)
        offsets_all = np.concatenate(region_offs)
        region_of_all = np.repeat(
            np.arange(len(region_names)), [len(x) for x in region_idx])
        dense_idx, first = np.unique(dense_idx_all, return_index=True)
        offsets = offsets_all[first]
        region_of_vertex = region_of_all[first]
        scalp_region_id = region_names.index("scalp")
        print(f"    {len(sk_points)} puncte pe craniu; constrangeri dense: "
              f"{len(scalp_idx)} scalp + "
              f"{len(dense_idx) - len(scalp_idx)} facial "
              f"({', '.join(f'{r[0]}:{len(r[1])}' for r in face_regions)})")
        # Sanity: markerii (piele = os + adancime) trebuie sa fie aproape de
        # suprafata craniului; altfel probabil spatii/unitati diferite.
        d_check, _ = tree.query(targets_xyz)
        if float(np.median(d_check)) > 30.0:
            warnings_list.append(
                f"Distanta mediana markeri->craniu este "
                f"{np.median(d_check):.1f} mm (>30) - posibil ca craniul sa "
                f"nu fie in acelasi spatiu/unitati ca CSV-ul!")
        dense = {
            "dense_idx": dense_idx, "offsets": offsets, "tree": tree,
            "points": sk_points, "normals": sk_normals,
            "region_names": region_names, "region_of_vertex": region_of_vertex,
            "scalp_region_id": scalp_region_id,
            "weight_ratio": cfg.dense_weight,
            # Respingere per-vertex: o corespondenta valida sta la ~offset mm
            # de os; peste offset+12 mm e sigur o zona lipsa (ex. mandibula
            # absenta -> barbia nu se lipeste de maxilar).
            "max_dists": offsets + 12.0,
            "min_dot": 0.2, "triangles": triangles, "flip": None,
            "in_fit": not cfg.no_dense_fit,
        }
        # Harta vertex->regiune pentru coloana "Regiune" din tabelul markeri.
        vertex_region = build_vertex_region_map(len(mu), scalp_idx,
                                                face_regions)
    else:
        vertex_region = None

    # --- Etapa 1: aliniere initiala (pentru raportare) --------------------
    print("[1] Aliniere Umeyama ponderata (robusta)...")
    try:
        s1, r1, t1, res_align = robust_alignment(mu[lm_idx], targets_xyz, weights)
    except ValueError as e:
        print(f"[EROARE FATALA] {e}")
        return 2
    print(f"    scala = {s1:.4f}, RMS = {res_align.mean():.2f} mm "
          f"(max {res_align.max():.2f} mm)")

    swapped = check_side_swap(targets, r1, t1, s1)
    if swapped:
        warnings_list.append(
            "Markeri suspecti de inversare Stanga/Dreapta: "
            + ", ".join(swapped) + ". Verifica plasarea in Blender.")

    # --- Etapa 2: fit statistic -------------------------------------------
    lam_arg = "auto" if str(cfg.regularization).lower() == "auto" else float(cfg.regularization)
    distance_pairs = ([(label_to_vertex[a], label_to_vertex[b])
                       for a, b in CONSISTENCY_PAIRS]
                      if cfg.distance_weight > 0.0 else None)
    active_terms = []
    if dense and dense["in_fit"]:
        active_terms.append("+ constrangeri dense")
    if loss_cfg.symmetry_weight > 0.0:
        active_terms.append("+ simetrie")
    if distance_pairs:
        active_terms.append("+ distante")
    if loss_cfg.prior_soft_sigma > 0.0:
        active_terms.append("+ prior moale")
    print(f"[2] Fit statistic (regularizare: {cfg.regularization}"
          f"{''.join(', ' + t for t in active_terms)})...")
    c, scale, rot, trans, lam_used, fit_info, res_fit = fit_identity(
        mu, basis, lm_idx, targets_xyz, weights, lam=lam_arg, dense=dense,
        mirror_indices=model.mirror_indices, distance_pairs=distance_pairs,
        loss_cfg=loss_cfg)
    print(f"    lambda = {lam_used:g}, |c|max = {np.abs(c).max():.2f} sigma, "
          f"RMS = {res_fit.mean():.2f} mm (max {res_fit.max():.2f} mm)")
    stab = stability_warnings(lam_used, fit_info[0], c, loss_cfg.clip_sigma)
    for w in stab:
        print(f"    [!] {w}")
    warnings_list.extend(stab)

    # Excludere automata a outlierilor (optional): markerii cu reziduu mare
    # sunt scosi COMPLET (nu doar down-ponderati) si fitul se reia o data;
    # ei nu devin nici centre TPS (ar injecta warp local direct in fata).
    excluded_auto = []
    if cfg.exclude_outliers:
        med = float(np.median(res_fit))
        mad = 1.4826 * float(np.median(np.abs(res_fit - med)))
        thresh_ex = max(15.0, 3.0 * mad)
        drop = res_fit > thresh_ex
        if drop.any():
            excluded_auto = [(labels[i], float(res_fit[i]))
                             for i in np.where(drop)[0]]
            keep = ~drop
            targets = [t for t, k in zip(targets, keep) if k]
            labels = [t[0] for t in targets]
            lm_idx = lm_idx[keep]
            targets_xyz = targets_xyz[keep]
            weights = weights[keep]
            res_align = res_align[keep]
            print(f"    [!] Exclusi automat {len(excluded_auto)} markeri "
                  f"(reziduu > {thresh_ex:.1f} mm): "
                  + ", ".join(f"{l} ({r:.1f} mm)" for l, r in excluded_auto))
            warnings_list.append(
                f"Markeri exclusi automat (--exclude-outliers), reziduu > "
                f"{thresh_ex:.1f} mm: "
                + ", ".join(f"{l} ({r:.1f} mm)" for l, r in excluded_auto)
                + ". Verifica plasarea lor in Blender.")
            if len(targets) < 4:
                print(f"[EROARE FATALA] Dupa excluderea outlierilor au ramas "
                      f"doar {len(targets)} markeri (minim 4).")
                return 2
            if len(targets) < 10:
                warnings_list.append(
                    f"Dupa excluderea outlierilor au ramas {len(targets)} "
                    f"markeri (<10) - reconstructie slab constransa.")
            print("    Refac fitul statistic fara outlieri...")
            c, scale, rot, trans, lam_used, fit_info, res_fit = fit_identity(
                mu, basis, lm_idx, targets_xyz, weights, lam=lam_arg,
                dense=dense, mirror_indices=model.mirror_indices,
                distance_pairs=distance_pairs, loss_cfg=loss_cfg)
            print(f"    refit: lambda = {lam_used:g}, |c|max = "
                  f"{np.abs(c).max():.2f} sigma, RMS = {res_fit.mean():.2f} mm "
                  f"(max {res_fit.max():.2f} mm)")
            stab = stability_warnings(lam_used, fit_info[0], c,
                                      loss_cfg.clip_sigma)
            for w in stab:
                print(f"    [!] {w}")
            warnings_list.extend(stab)

    outliers, thresh = flag_outliers(labels, res_fit)
    if outliers:
        warnings_list.append(
            f"Markeri cu reziduu mare dupa fit (prag {thresh:.1f} mm) - "
            "verifica plasarea si adancimea tesutului: "
            + ", ".join(f"{l} ({r:.1f} mm)" for l, r in outliers))

    # --- Mesh-ul in spatiul world (mm, Blender) ----------------------------
    v_model = model.generate(c)
    v_world = scale * (v_model @ rot.T) + trans
    lm_world = v_world[lm_idx]

    # --- Etapa 3: corectie locala limitata --------------------------------
    dense_report = None
    if dense is not None:
        _, _, dense_stats = fit_info
        dense_report = [
            f"Constrangeri dense: offset scalp = {cfg.scalp_offset_mm:g} mm"
            + ("" if cfg.no_face_dense else
               ", facial: " + ", ".join(
                   f"{k}={v:g}" for k, v in REGION_OFFSETS_MM.items())
               + " mm"),
            f"  pondere totala = {cfg.dense_weight:g} x markeri, "
            f"in fit = {dense['in_fit']}, in TPS = {not cfg.no_dense_tps}",
        ]
        if dense_stats:
            first, last = dense_stats[0], dense_stats[-1]
            dense_report.append(
                f"  Corespondente pastrate: prima iter. {first[0]}"
                f" (dist. medie {first[1]:.2f} mm) -> ultima iter. {last[0]}"
                f" (dist. medie {last[1]:.2f} mm)")

    # Cap de corectie per-vertex: scalpul (tesut subtire, forma sigura) poate
    # fi tras pana la --max-correction-mm; fata (ochi/nas/gura, geometrie
    # fina) este limitata la --face-cap-mm. Masca de scalp e disponibila din
    # model, deci cap-urile functioneaza si fara --skull.
    scalp_caps = (scalp_idx if cfg.skull else
                  build_scalp_mask(mu, vertex_groups, vertex_group_names))
    cap_vertex = np.full(len(mu), cfg.face_cap_mm)
    cap_vertex[scalp_caps] = cfg.max_correction_mm
    protected_idx = build_protected_mask(vertex_groups, vertex_group_names)

    if cfg.skip_tps:
        v_final = v_world
        field = np.zeros_like(v_world)
        clamped = np.zeros(len(targets))
        n_scalp_centres = n_face_centres = 0
    else:
        centers = [lm_world]
        # Reziduurile markerilor se scaleaza cu increderea: un marker
        # imprecis (ex. Supraorbitale 0.5) devine constrangere PARTIALA,
        # nu trage mesh-ul pana la tinta lui (posibil gresita).
        res_vecs = [(targets_xyz - lm_world) * weights[:, None]]
        n_scalp_centres = n_face_centres = 0
        if dense is not None and not cfg.no_dense_tps:
            from scipy.spatial import cKDTree
            sidx, dt_w, keep, _ = dense_correspondences(v_world, dense)
            kept_regions = dense["region_of_vertex"][keep]
            if keep.sum() > 0:
                # Deduplicare: centrele dense aflate aproape de un marker
                # (ex. Vertex/Eurion sunt in masca de scalp) ar face matricea
                # TPS aproape singulara (valori usor conflictuale -> camp
                # oscilant). Markerii au prioritate, centrele se filtreaza.
                d_to_lm, _ = cKDTree(lm_world).query(v_world[sidx])
                far = d_to_lm > 3.0
                sidx, dt_w = sidx[far], dt_w[far]
                kept_regions = kept_regions[far]
            rng = np.random.default_rng(42)
            # Centre scalp si faciale, sub-esantionate separat.
            scalp_id = dense["scalp_region_id"]
            for is_scalp, cap_n in ((True, cfg.tps_scalp_centres),
                                    (False, cfg.tps_face_centres)):
                sel_mask = (kept_regions == scalp_id) if is_scalp else (
                    kept_regions != scalp_id)
                cand = np.where(sel_mask)[0]
                if len(cand) == 0:
                    continue
                n_sub = min(cap_n, len(cand))
                # Indici sortati => perechile (vertex, tinta) raman aliniate.
                sel = np.sort(rng.choice(cand, size=n_sub, replace=False))
                centers.append(v_world[sidx[sel]])
                res_vecs.append(dt_w[sel] - v_world[sidx[sel]])
                if is_scalp:
                    n_scalp_centres = len(sel)
                else:
                    n_face_centres = len(sel)
        centers = np.vstack(centers)
        res_vecs = np.vstack(res_vecs)

        print(f"[3] Corectie locala TPS (cap fata {cfg.face_cap_mm:g} mm / "
              f"scalp {cfg.max_correction_mm:g} mm, "
              f"damping zone protejate x{cfg.protect_damping:g}, "
              f"{len(targets)} markeri + {n_scalp_centres} scalp + "
              f"{n_face_centres} faciale)...")
        v_final, field, clamped = bounded_tps_correction(
            v_world, centers, res_vecs, cap_vertex,
            protected_idx=protected_idx,
            protect_damping=cfg.protect_damping)
        mean_corr = float(np.linalg.norm(field, axis=1).mean())
        max_corr = float(np.linalg.norm(field, axis=1).max())
        print(f"    corectie: medie {mean_corr:.2f} mm, max {max_corr:.2f} mm")
        if mean_corr > 5.0:
            warnings_list.append(
                f"Corectia locala medie este mare ({mean_corr:.1f} mm) - "
                "posibil markeri plasati inconsistent; verifica raportul.")

    res_final = np.linalg.norm(v_final[lm_idx] - targets_xyz, axis=1)

    # Metrici finale per regiune: distanta la craniu (tinta = offsetul
    # regiunii) si corectia medie aplicata.
    if dense is not None:
        field_mag = np.linalg.norm(field, axis=1)
        sidx_f, _, keep_f, _ = dense_correspondences(v_final, dense)
        reg_f = dense["region_of_vertex"][keep_f]
        dense_report.append("  Per regiune (final): pastrate, distanta "
                            "medie la craniu [tinta], corectie medie:")
        off_all = dense["offsets"]
        for rid, rname in enumerate(dense["region_names"]):
            m = reg_f == rid
            if not m.any():
                dense_report.append(f"    {rname:14s}: 0 corespondente")
                continue
            vids = sidx_f[m]
            d_r, _ = dense["tree"].query(v_final[vids])
            d_r = np.minimum(d_r, 60.0)  # taie cozile (zone fara os)
            target_off = off_all[keep_f][m].mean()
            dense_report.append(
                f"    {rname:14s}: {m.sum():5d} | {d_r.mean():6.2f} mm "
                f"[{target_off:4.1f}] | {field_mag[vids].mean():5.2f} mm")
            print(f"    {rname:14s}: dist. la craniu {d_r.mean():6.2f} mm "
                  f"(tinta {target_off:4.1f})")

    # --- Etapa 4: export ---------------------------------------------------
    print(f"[4] Export OBJ: {cfg.output}")
    export_obj(cfg.output, v_final, triangles)
    print(f"    Export heatmap PLY: {cfg.output_error_mesh}")
    export_heatmap_ply(cfg.output_error_mesh, v_final, triangles,
                       np.linalg.norm(field, axis=1))

    extra = None
    extra_parts = []
    if cfg.skip_tps:
        extra_parts.append("Corectia locala a fost dezactivata (--skip-tps).")
    if loss_cfg.symmetry_weight > 0.0:
        extra_parts.append(
            f"Termen de simetrie bilaterala activ "
            f"(symmetry_weight = {loss_cfg.symmetry_weight:g}).")
    if distance_pairs:
        extra_parts.append(
            f"Termen de distante inter-landmark activ "
            f"(distance_weight = {loss_cfg.distance_weight:g}, "
            f"{len(distance_pairs)} perechi).")
    if loss_cfg.prior_soft_sigma > 0.0:
        extra_parts.append(
            f"Prior latent moale activ peste +/-{loss_cfg.prior_soft_sigma:g} "
            f"sigma (prior_soft_weight = {loss_cfg.prior_soft_weight:g}).")
    if extra_parts:
        extra = "\n".join(extra_parts)
    lm_regions = None
    if vertex_region is not None:
        lm_regions = {label: (vertex_region[vid] or "-")
                      for label, vid, _, _ in targets}
    write_stats(cfg.output_stats, cfg, targets, skipped, scale, lam_used,
                fit_info, res_align, res_fit, res_final,
                clamped[:len(targets)], field, c, warnings_list, extra,
                dense_report=dense_report, lm_regions=lm_regions,
                consistency=cons_rows, excluded_auto=excluded_auto)
    print(f"    Statistici: {cfg.output_stats}")
    print("Finalizat.")
    return 0
