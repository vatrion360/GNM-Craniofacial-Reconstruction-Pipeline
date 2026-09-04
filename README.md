## GNM Craniofacial Approximation — Blender Addon + Forensic Pipeline ##

_Forensic Anthropology · Facial Approximation · GNM Head v3.0_

Blender addon with **parallel dual viewports** and _live_ fitting of the Google GNM Head statistical model onto scanned skulls, plus a complete offline pipeline (Umeyama alignment · ridge regression with IRLS · TPS correction) for publication-grade reconstruction.

![Apache 2.0 License](https://img.shields.io/badge/license-Apache%202.0-blue) ![Python ≥ 3.10](https://img.shields.io/badge/python-%E2%89%A53.10-blue) ![Blender 3.6 – 5.0](https://img.shields.io/badge/blender-3.6%20%E2%80%93%205.0-orange) ![numpy 1.26+](https://img.shields.io/badge/numpy-1.26%2B-informational) ![model GNM Head v3.0](https://img.shields.io/badge/model-GNM%20Head%20v3.0-yellow) ![Blender tests 57/57](https://img.shields.io/badge/Blender%20tests-57%2F57%20PASS-brightgreen) ![pytest cranio 46/47](https://img.shields.io/badge/pytest%20cranio-46%2F47-yellowgreen) ![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20cross--platform-lightgrey)

## Contents

1. [About the project](#about-the-project)
2. [Components](#components)
3. [Requirements](#requirements)
4. [Installation](#installation)
    - 4.1 [Blender workflow (live)](#41-the-blender-addon)
    - 4.2 [Offline workflow (CLI)](#42-offline-workflow-cli)
5. [Options — Blender addon](#options--blender-addon)
6. [Options — gnm_reconstruct.py](#options--gnm_reconstructpy)
7. [Auxiliary scripts](#auxiliary-scripts)
8. [Landmarks & correspondences](#landmarks--correspondences)
9. [Architecture & methodology](#architecture--methodology)
10. [Testing](#testing)
11. [Known limitations](#known-limitations)
12. [References](#references)
13. [License](#license)

## About the project

A scientific tool for **forensic facial approximation** on scanned skulls, built on top of the **Google GNM Head v3.0** statistical head model (17,821 vertices, 253 identity components, fixed topology). The workflow has two stages:

### A. Blender addon (V13) — live preview

- Two parallel 3D viewports in the same window: **left** = skull + markers, **right** = GNM head regenerated from the β coefficients.
- Automatic re-fitting on every marker add/move/delete (worker thread + timer, responsive UI).
- Rigid alignment with scale (weighted Umeyama) + ridge regression with **adaptive** or **LOO-CV** regularization (like offline).
- Optional **demographic prior** (sex × ethnicity, from the GNM IdentitySampler, precomputed offline).
- **Markerless ICP alignment** (V13.2): the undeformed head "wraps" the skull at the push of a button (multi-start), then **skull morphology drives the general deformation** (dense constraints on scalp + thin-tissue regions, like `--skull` offline).
- **Fitting quality (V13.3–V13.5):** bony-bridge nasal constraint, full-weight far correspondences, 50/50 face/scalp row budget, rejection schedule, 10% outlier trim, per-region diagnostics in the status line.
- Confidence-colored ghosts at GNM landmark positions; manual vertex picking on the GNM mesh.

### B. Offline pipeline — final result

- Reads the marker CSV exported from the addon and produces **OBJ + PLY heatmap + TXT report**.
- Statistical fit with λ chosen by **LOO-CV**, Huber weights (IRLS), hard ±3σ clip, reflection detection (swapped L/R markers).
- Optional dense skull→skin constraints (scalp + thin-tissue regions) with outlier rejection.
- Local **TPS** correction (Bookstein 1989) with smooth per-vertex caps and protection on eyes/lips/mouth.
- Marker placement consistency checks _before_ the fit.

> **Important:** both stages use **exactly the same fitting math** (the `cranio.optimize.fit_identity` module, pure numpy). The Blender live preview is intentionally "within ~1 cm" of the targets — the difference is closed by the TPS stage, available only offline (scipy does not ship with Blender).

## Components

| File / directory                            | Role                                                                                                                                                                                                                                                                                 |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `addon_v13.py`                              | The complete Blender addon (markers + dual viewports + live fitting + demographic prior + ICP/dense deformation). English UI.                                                                                                                                                        |
| `addon_v12.py`                              | Previous version, kept for rollback (markers only, no live mode).                                                                                                                                                                                                                    |
| `gnm_reconstruct.py`                        | The offline pipeline CLI (thin wrapper over `cranio.pipeline`).                                                                                                                                                                                                                      |
| `cranio/`                                   | The pure-Python numeric package: `backend` (npz loading + anatomically verified landmark↔vertex tables), `optimize` (Umeyama, ridge, IRLS, LOO, TPS, prior, huber_rows/pose_rows), `landmarks` (registry + weights), `io_csv`, `checks`, `geometry`, `pipeline`, `export`, `report`. |
| `make_demographic_prior.py`                 | Generates `priors/prior_<SEX>_<ETHNICITY>.npz` from the IdentitySampler (requires TensorFlow; runs in the venv, _not_ in Blender).                                                                                                                                                   |
| `build_landmark_map.py`                     | Regenerates `landmark_vertex_map.json` from the 68 official GNM landmarks + manually verified overrides (V13.4).                                                                                                                                                                     |
| `landmark_vertex_map.json`                  | The landmark→vertex GNM map with `source`/`confidence` (18 entries, reviewed V13.4; manual Blender picks merge here).                                                                                                                                                                |
| `priors/`                                   | The 8 demographic prior files (2 sexes × 4 ethnicities; μ and σ per β component).                                                                                                                                                                                                    |


## Requirements

### Blender addon (live)

- **Blender ≥ 3.6** (developed and tested on **Blender 5.0.1**, Windows; the code is cross-platform).
- Only **numpy** — already bundled with Blender (1.26+ on 5.0). scipy/TensorFlow are _not_ needed in Blender.
- The `gnm_head.npz` file (in the repo), the `cranio/` package (included), `landmark_vertex_map.json`, optionally `priors/`.
- ~1 GB free RAM for the identity basis (loaded as float32, ~540 MB).

### Offline pipeline (CPython)

- **Python ≥ 3.10** (the repo ships a `.venv` with 3.12).
- `numpy`, `scipy`, `trimesh`.
- `tensorflow` — **only** for `make_demographic_prior.py` and the GNM demos (not for reconstruction).
- pytest — only to run the test suites.

## Installation

### 4.0. The offline pipeline (one time)

```
# in the repo root
python -m venv .venv
.venv\Scripts\activate              # Windows; on Linux: source .venv/bin/activate
pip install numpy scipy trimesh
pip install tensorflow              # optional: only for prior generation
.venv\Scripts\python.exe make_demographic_prior.py   # generates priors/*.npz (8 combos)
```

### 4.1. The Blender addon

1. **Install:** Blender → `Edit ▸ Preferences ▸ Add-ons ▸ Install from Disk…` → select `addon_v13.py` → enable _"GNM Scientific Markers"_. Alternative: Scripting → Text Editor → Open → Run Script (for development; F8 = reload).
2. **Data paths:** if `addon_v13.py` lives in the repo root, the paths to `gnm_head.npz`, `landmark_vertex_map.json` and `priors/` are **auto-detected**. Otherwise set them manually in the panel (V13 section).
3. The panel lives in `3D View ▸ Sidebar (N) ▸ GNM Markers`.

#### Step-by-step live workflow

1. **"1. Import & Calibrate Skull"** — imports .stl/.obj, joins multiple objects, centers, scales to mm, fixes normals.
2. **"2. Load Marker List"** — creates the 27 entries (24 + the 3 nasal V13.6 landmarks: Acanthion, Piriform Dr/St; with literature tissue depths).
3. **"Setup / Repair Layout"** (V13 section) — splits the area into two viewports: left = skull, right = GNM (per-area local-view, same world). Re-run anytime to "repair".
4. **"Load Model & Map"** — loads the npz (float32, mm) and the JSON; creates the GNM mesh + ghosts in the right viewport.
5. **"Start Live"** — starts the fitting worker.
6. Optional (V13.2), with the skull selected: **"Prepare Skull"** → **"Align & Deform from Skull (ICP)"** — the undeformed head aligns to the skull _with zero markers_ (~2-4 s, visible progress), and skull morphology drives the general deformation.
7. **Place markers** with "Place Marker" (click on the skull on the left; auto-advances to the next). From **3 markers** the GNM head starts aligning and deforming in real time.
8. Optional: **Sex + Ethnicity** (demographic prior), the **"Deformation"** slider, **"λ auto (LOO)"**, **"Continuous Dense"** (skull constraints at every fit), **"Dense Strength"** / **"Dense Nose Weight"** sliders, manual picking **"Pick GNM Vertex"** on the right viewport, GNM overlay on the skull (👁 button).
9. **"4. Export Final CSV"** — for the publication-grade offline reconstruction.

### 4.2. Offline workflow (CLI)

```
# full reconstruction: statistical fit + dense skull constraints + TPS
.venv\Scripts\python.exe gnm_reconstruct.py --input skull.csv --skull skull.stl --regularization auto --exclude-outliers

# outputs (defaults, next to the input):
#   skull_reconstructie.obj             - final mesh (mm, Blender world)
#   skull_reconstructie_heatmap.ply     - per-vertex map of the local correction
#   skull_reconstructie_statistici.txt  - full report (λ, RMS, residuals, warnings)
```

## Options — Blender addon

### 5.1. Markers section (V12, unchanged)

|Control|Description|
|---|---|
|Marker Radius / Peg Thickness (mm)|Visual sizes of the empties and tissue-depth "pegs".|
|Marker list (UIList)|27 landmarks (24 + 3 nasal V13.6) with placed/unplaced status, side [M]/[R]/[L], editable tissue depth (mm) and a GNM correspondence status icon (🔗 mapped, 📌 manual, ? low confidence, ! no correspondence).|
|Place Marker|Modal: click on the skull → creates an empty on the bone + a skin target (bone + depth along the normal) + a "peg"; re-placement replaces (= move). The raycast skips GNM objects (which wrap the skull).|
|Next Unplaced / Recenter / Plane / Asymmetry Report / Mirror Reconstruction|The V12 toolset for partial skulls: midsagittal plane from midline markers, bilateral asymmetry report, welded mirror reconstruction, weld-free preview.|
|Session Report / Standardized Captures / Export CSV|Documentation: Text Editor report, 3 orthographic captures (front/3-4/profile) with fixed camera+light relative to the skull axes, v11/v12-compatible CSV export.|

### 5.2. "GNM Live Reconstruction (V13)" section

|Control|Default|Description|
|---|---|---|
|Setup / Repair Layout|—|Vertical `area_split` + per-area local-view (left: everything except GNM; right: GNM only). Idempotent — rebuilds both local views.|
|👁 Overlay|off|Shows/hides the GNM objects in the left viewport too (direct overlay on the skull — possible thanks to the shared world).|
|Path gnm_head.npz / landmark_vertex_map.json|auto|Data paths; auto-detected from the addon location.|
|Load Model & Map|—|Loads the model (float32) + the JSON; creates the GNM mesh (once; re-loading resets it) + the ghosts.|
|Start / Stop Live|off|Starts/stops worker thread + timer (persistent) + depsgraph handler. Clean shutdown on F8/disable.|
|Show Ghosts|on|Overlay of confidence-colored empties at GNM landmark positions: green = V12/JSON safe, orange = JSON low, purple = manual pick, red = no correspondence, cyan = Gerasimov pronasale estimate (V13.6 diagnostic). The same colors are applied to the placed marker objects (bone empty, skin target and peg) in the left viewport.|
|Deformation (lambda base) `V13.1`|**1.0**|Regularization at the full marker set: `λ = base · 24 / n_markers`, clamped to [λ_min, λ_max]. Calibrated on real data: 1.0 ≈ the offline robust morphology; ↑ = more "average"/rigid head, ↓ towards 0.3 = maximum deformation.|
|λ auto (LOO, like offline) `V13.1`|off|When the _marker count_ changes (n≥4), λ is chosen by LOO-CV (like the offline pipeline; ~1 s once, in a thread). Effective: `max(λ_LOO, 0.3·24/n)`.|
|Sex (prior) `V13.1`|Unknown|Unknown / Female / Male — activates the demographic prior (shrink towards the sex mean from IdentitySampler).|
|Ethnicity (prior) `V13.1`|Unknown|Unknown / Middle Eastern / Asian / White-Caucasian / Black-African (the GNM CVAE categories).|
|Prior Strength `V13.1`|1.0|Multiplier of the prior precision (0.1–4.0); 1.0 = exact Gaussian MAP.|
|Prepare Skull `V13.2`|—|Samples the skull surface (active object): ~60k world points + numpy normals + KD-tree. Required for ICP and the dense constraints.|
|Align & Deform from Skull (ICP) `V13.2`|—|One-shot (~2-4 s): multi-start ICP (yaw 0/90/180/270, coarse + refine) aligns the _undeformed_ head to the skull even with 0 markers, then 6 dense fits drive the general deformation from skull morphology. V13.3–V13.5: rejection schedule 2.0→1.0, relaxed normal test, 50/50 row budget, 10% farthest-correspondence trim.|
|Continuous Dense `V13.2`|off|Dense constraints (scalp + thin-tissue regions, with offsets and offline-style rejection) at every fit; rate drops to ~2-3 Hz. Markers remain the primary anchors; λ is computed from markers only.|
|Scalp only (conservative) `V13.2`|off|Dense constraints on the scalp only (equivalent to `--no-face-dense`); recommended for skulls with a damaged face.|
|Dense Strength `V13.3`|**1.0**|Multiplier of the ICP/dense attraction force (0.1–10). Increase if skull areas remain uncovered; decrease if the head sticks too aggressively to the bone.|
|Dense Nose Weight `V13.3`|**0.7**|Weight of the dense constraint on the nasal bridge, as a "soft prior": below 1.0 the nasal landmarks (Nasion/Rhinion/Nasospinale/Alare) and the statistical model decide the nose shape; 0 = no dense nose constraint.|
|Pick GNM Vertex|—|Modal picking on the GNM mesh (click on the right) → manual vertex override for the active marker (useful for landmarks without a correspondence or fine corrections).|
|✕ (clear override)|—|Clears the active marker's manual override (back to the V12 > JSON chain).|
|🗑 (delete)|—|Deletes the active marker's empties (moving = re-placing).|
|Export Updated JSON Map|—|Merges manual picks into `landmark_vertex_map.json` (`source="manual_picked_blender"`, `.bak` backup).|
|Advanced: Update Rate (Hz)|10|Result applications per second (the fit itself runs in the worker, latest-wins).|
|Advanced: Lambda Min / Max|0.3 / 1000|Adaptive regularization bounds (λ_min = the offline LOO grid edge).|
|Advanced: Max Dense Rows `V13.3`|1500|Maximum dense rows per fit (50/50 face/scalp budget); more = denser constraint (better coverage), slower fit.|
|Advanced: Clip Sigma `V13.3`|3.0|Hard limit of identity coefficients (±σ). WARNING: above 3.0 the head can leave the model's plausible domain — watch the `clip N` indicator.|
|Advanced: Priors Folder|`<repo>/priors`|Location of the `prior_<SEX>_<ETHNICITY>.npz` files.|

> **The live status line** shows, at every fit: marker count, λ used and its source (`formula`/`loo`), RMS, max residual + its landmark, **‖β‖** and **the number of components at the ±3σ clip** (model "tension" indicator). The dense line adds correspondences kept + mean distance, plus the **nasal diagnostics (V13.3)**: bridge keep-rate, mean distance to bone vs the 3 mm target, and a separate RMS over the nasal landmarks.

## Options — `gnm_reconstruct.py`

All CLI options, grouped by pipeline stage. Usage: `python gnm_reconstruct.py --input markers.csv [options]`.

### 6.1. Input / output

|Option|Default|Description|
|---|---|---|
|`--input` **required**|—|CSV exported from the addon (v11/v12/v2; `0,0,0` rows = unplaced markers, ignored).|
|`--output`|`<input>_reconstructie.obj`|Output OBJ (vertices in mm, the scene's Blender world).|
|`--output-error-mesh`|`<output>_heatmap.ply`|PLY with a per-vertex colormap of the local (TPS) correction magnitude.|
|`--output-stats`|`<output>_statistici.txt`|Full TXT report: λ, RMS, per-landmark residuals, history, stability warnings.|
|`--npz`|`gnm/shape/data/versions/v3_0/gnm_head.npz`|Path to the GNM Head model weights.|

### 6.2. Statistical fit (stage 2)

|Option|Default|Description|
|---|---|---|
|`--regularization`|`auto`|`auto` = λ chosen by leave-one-out cross-validation (grid 0.3–1000), or a fixed value (e.g. `30`).|
|`--exclude LABEL…`|—|Marker labels manually excluded from the fit _and_ from TPS centers (e.g. `--exclude Pogonion Rhinion`).|
|`--exclude-outliers`|off|Automatically excludes markers with residual > max(15 mm, 3·MAD) after the first fit and re-fits once; by default they are only flagged.|

### 6.3. Dense skull→skin constraints (stage 0b/2, with `--skull`)

|Option|Default|Description|
|---|---|---|
|`--skull`|—|The skull (STL/OBJ) in the same Blender scene as the CSV (world, mm). Enables dense constraints: scalp + thin-tissue facial regions (brow ~5 mm, nasal bridge ~3, zygomatic ~8.5, infraorbital ~7, chin ~10), with automatic rejection of invalid correspondences.|
|`--scalp-offset-mm`|5.0|Soft-tissue skin–bone offset on the scalp (mm).|
|`--dense-weight`|0.5|Total weight of the dense constraints, relative to the sum of marker weights.|
|`--dense-samples`|200000|Number of points sampled on the skull surface (KD-tree).|
|`--no-face-dense`|off|Dense constraints on the scalp only (like v2), not on the thin-tissue facial regions.|
|`--no-dense-fit`|off|No dense constraints in the statistical fit (they remain only in TPS).|

### 6.4. Local TPS correction (stage 3)

|Option|Default|Description|
|---|---|---|
|`--skip-tps`|off|Stops after the statistical fit (no local correction) — equivalent to live-preview quality.|
|`--max-correction-mm`|15.0|Smooth (tanh) per-vertex correction cap on the **scalp** (mm).|
|`--face-cap-mm`|8.0|Smooth per-vertex correction cap on the **face** (eyes/nose/mouth) (mm).|
|`--protect-damping`|0.25|Damping of the TPS correction on regions without anatomic anchors (eyes/mouth interior/lips); 1.0 = no protection.|
|`--tps-scalp-centres`|500|Max number of scalp points used as TPS centers.|
|`--tps-face-centres`|200|Max number of facial points used as TPS centers.|
|`--no-dense-tps`|off|No dense centers in the TPS correction (markers only).|

### 6.5. Optional loss terms (all off by default = results identical to v3.1)

|Option|Default|Description|
|---|---|---|
|`--symmetry-weight`|0.0|Weight of the bilateral symmetry prior in latent space (relative to λ); 0 = inactive.|
|`--distance-weight`|0.0|Total weight of inter-landmark distance constraints (target = the statistical template), relative to the sum of marker weights; 0 = inactive.|
|`--prior-soft-sigma`|0.0|Threshold (σ) beyond which the _soft_ latent prior activates (instead of the single hard ±3σ clip); 0 = inactive.|
|`--prior-soft-weight`|4.0|Strength of the soft prior (multiple of λ).|

> **Note (V13.1–V13.3):** `cranio.optimize.fit_identity` also accepts `prior_mean` / `prior_scale` / `prior_weight` (Gaussian demographic prior, MAP) and, for the Blender live path, `huber_rows` (Huber IRLS only on the leading marker rows — dense rows keep full weight) and `pose_rows` (the similarity transform is re-estimated from markers only, exactly like the offline pipeline's dense mode). The offline CLI exposes none of these (behavior unchanged, bit-identical without them).

## Auxiliary scripts

### 7.1. `make_demographic_prior.py` — generates the demographic priors

Samples N identities from the `IdentitySampler` (sex×ethnicity conditioned CVAE; requires TensorFlow — runs in the venv, **not** in Blender) and saves μ and σ per β component to `priors/prior_<SEX>_<ETHNICITY>.npz`. σ is clipped to [0.25, 4.0].

|Option|Default|Description|
|---|---|---|
|`--samples`|512|Number of identities sampled per combo.|
|`--out`|`<repo>/priors`|Output directory.|
|`--only SEX ETHNICITY`|all 8|Generates a single combo, e.g. `--only MALE WHITE`.|
|`--seed`|42|RNG seed (reproducibility).|

```
.venv\Scripts\python.exe make_demographic_prior.py                 # all 8 combos
.venv\Scripts\python.exe make_demographic_prior.py --only MALE WHITE --samples 1024
```

### 7.2. `build_landmark_map.py` — regenerates the landmark↔vertex map

Rebuilds `landmark_vertex_map.json` from (a) the 68 official GNM landmarks (`GNMLandmarksType.HEAD_SPARSE_68`, barycentric → nearest vertex, iBUG/dlib-68 compatible order) and (b) manually verified overrides (V13.4: Rhinion from iBUG 29, corrected glabella/gonion, verified zygion). Each entry gets `source` and `confidence`; manual Blender picks ("Pick GNM Vertex" + "Export Updated JSON Map") merge into and refine the map.

## Landmarks & correspondences

Working set: **27 individually verified craniometric landmarks** (no interpolated contour points): 9 midline (Nasion, Rhinion, Glabella, Pogonion, Gnathion, Vertex_VarfCap, Nasospinale_BazaNas, Prosthion_BuzaSup, **Acanthion** — V13.6) and 9 bilateral pairs (Gonion, Orbita Ext/Int, Supraorbitale, Infraorbitale, Zygion, Alare, Eurion, **Piriform** — V13.6). Soft-tissue depths come from the literature (Rhine & Campbell 1980; De Greef et al. 2006; Stephan & Simpson 2008). The V13.6 nasal landmarks (skin projection of the anterior nasal spine, 12297, and of the inferior piriform-aperture margin, 10215/4087 — exact topological mirrors) feed the Gerasimov nose-projection diagnostic below and participate in the fit as regular markers.

**The landmark → GNM vertex precedence chain (V13)**

1. **Manual override** — picking on the GNM mesh in the right viewport (📌, purple).
2. **The anatomically verified V12 table** (`cranio.backend.LABEL_TO_VERTEX`) — covers all 27; corrects the old values (e.g. Gonion 8737/2609, not the iBUG 0/16 points near the ears). **V13.4:** Rhinion = **12310** (iBUG 29, bony nasal bridge, 25.1 mm below Nasion) — the old 12296 is iBUG 30 = pronasale (nose tip), kept as a separate `pronasale` entry in the JSON. **V13.6:** Acanthion = 12297, Piriform Dr/St = 10215/4087 (geometric candidates from `tools/suggest_nasal_vertices.py`, visually verified).
3. **`landmark_vertex_map.json`** — source of names and _confidence_ (colors the warnings; green = safe, orange = low).

Landmarks with no candidate appear in the list with **!** ("needs manual GNM-side picking") and are excluded from the fit until manually picked.

### Nose-projection diagnostic, Gerasimov / Ullrich & Stephan 2011 (V13.6)

When Nasion, Rhinion, Acanthion and both Piriform points are placed, the pipeline estimates the pronasale at the intersection of two tangents in the mid-sagittal plane: the _upper_ tangent at the distal end of the nasal bones (last ~2 mm towards Rhinion, from the skull profile; with an anisotropy guard — if the 2 mm patch is near-isotropic, it falls back to the Nasion→Rhinion direction) and the _lower_ tangent through Acanthion and the two inferior piriform-aperture points (the Ullrich & Stephan 2011 reading — the aperture rim, _not_ the nasal spine's own direction).

**Outputs (purely informational — never touches `fit_identity`):** a block in the TXT report (tangent angle, estimated position, distance to Rhinion as a sanity metric, deviation vs the final fitted pronasale, vertex 12296) and, in the live addon, a separate cyan ghost + a status line next to the V13.3 nasal diagnostics — computed in the worker thread (pure numpy, the same `cranio.geometry.gerasimov_pronasale` in both surfaces).

**Validation (ken-13test, both surfaces):** the estimate deviated **14.5–22.1 mm** from the fitted pronasale with a consistent sign (posterior-inferior), well above the ~5 mm forensic tolerance. **Decision: kept as a diagnostic only** — it was _not_ promoted to a weighted soft-prior in the fit (a method with this accuracy must not pull β). It remains useful as a cross-check for marker placement and nose morphology.

> **Orientation/units:** the GNM npz is in **meters** (+X = anatomic left, +Y = up, +Z = anterior); the addon/pipeline work in **mm**. The fit is fully frame-agnostic (Umeyama recovers rotation+scale+translation), so the skull can have any orientation in the scene. The fit target = the _skin_ position (bone + tissue depth), identical to the CSV export.

## Architecture & methodology

### The offline pipeline (4 stages)

1. **Loading + checks (Stage 0):** CSV reading, minimum 4 markers, inter-landmark distance ratios CSV vs template (catches misplaced markers _before_ the fit), optional dense skull constraints.
2. **Similarity alignment (Stage 1):** weighted, robust Umeyama 1991 (Huber, k=10 mm); **reflection (det=−1) raises an error** — flags swapped Left/Right markers.
3. **Statistical fit (Stage 2):** alternating estimation of β ↔ similarity transform: weighted ridge LSQ in augmented primal form (`lstsq` SVD), Huber IRLS weights, hard ±3σ clip, λ by LOO-CV (`auto`) with grid-edge stability warnings. Pluggable terms: dense constraints, symmetry, distances, soft prior, demographic prior (V13.1, live only).
4. **Bounded local correction (Stage 3):** TPS field (Bookstein 1989, `scipy RBFInterpolator`) with smooth tanh per-vertex caps (scalp 15 mm / face 8 mm) and damping on unanchored regions.
5. **Export (Stage 4):** OBJ + PLY heatmap + TXT with complete statistics.

### The Blender live preview (V13)

- **Event → fit:** `depsgraph_update_post` with a fingerprint over marker positions (no constant polling; catches add/move/delete/undo) → "latest-wins" snapshot to the worker.
- **Worker thread (numpy only, zero bpy):** the same `fit_identity` as offline (partial fitting on any subset ≥3), adaptive λ `clamp(1.0·24/n, 0.3, 1000)` or LOO-CV cached per marker count; ~30–120 ms/fit.
- **Application (main thread, ~10 Hz):** `bpy.app.timers` rewrites only the vertex positions of the existing object (`foreach_set`; the object is not recreated), updates `matrix_world` (T·S·R) and the ghosts.
- **Model:** `V = μ + Σ βᵢ·Bᵢ` (linear in β at zero expression; equivalent to `gnm(identity=β)`), loaded directly from the npz as float32 (~540 MB).

### ICP alignment + general deformation from the skull (V13.2–V13.5)

- **Preparation:** the skull is sampled from the mesh (~60k world points + triangle normals, numpy) and indexed with `mathutils.kdtree` (queried read-only from the worker; no scipy/trimesh).
- **Multi-start ICP (coarse stage, 0+ markers):** yaw hypotheses (0/90/180/270°; the skull is roughly upright after import), initialization on consistent sets (bbox + centroid over the whole mesh), decreasing acceptance radius (4×→1× of offset+12 mm — widens the attraction basin), coarse round + long refine on the winner; score = mean distance / coverage (rejects degenerate minima). ~0.4 s.
- **General deformation (stage B):** the dense correspondences (scalp ~2280 + 5 thin-tissue regions, with 3–10 mm offsets and distance+normal rejection — exactly like offline) become "pseudo-marker" rows in the same `fit_identity`: **deformation flows through β (GNM space + ±3σ clip + prior), so the head stays guaranteed plausible** — no free warp. Correspondences refresh at snapshot level (outer-ICP), with visible progress.
- **Fitting quality (V13.3):** the nasal-bridge patch is centered on the _bony_ bridge (the old patch was centered on the pronasale vertex and flattened the nose); dense rows keep full weight (no Huber — the farthest points, which need the strongest pull, are no longer penalized); the similarity pose is re-estimated from markers only (`pose_rows` — the dense mass no longer drags the global scale); 50/50 face/scalp row budget (the scalp, ~73% of vertices, no longer drowns the facial regions); rejection schedule 2.0→1.0 in the job.
- **Anti-saturation (V13.5):** at every job iteration the farthest 10% of correspondences are trimmed — the distribution tail is dominated by wrong matches (inner skull table, aperture rims) that pushed β to saturation ("exaggerated face"); the normal test stays on but relaxed (dot > −0.2, only opposite-side matches are rejected). Measured on v13-test: clip 30→23 components, bone exposed through skin 4.8%→2.0%.
- **Partial skulls:** the rejection (distance > offset+12 mm or nearly opposite normal) automatically excludes missing regions (e.g. the chin with an absent mandible); GNM face orientation is determined deterministically from the signed volume.

## Testing

```
# addon suite (71 tests: registration, math, scene, live services, prior, λ/LOO, dense/ICP, Gerasimov)
blender -b --factory-startup -P test_addon_v13_headless.py

# cranio package suite (66 tests: backend, fit, Umeyama, TPS, IO, stability, dense regions, geometry)
.venv\Scripts\python.exe -m pytest tests -q

# real-case verification harnesses (v13-test/; read-only, print quality metrics)
.venv\Scripts\python.exe v13-test\verify_fit_quality.py    # marker-workflow quality gates
.venv\Scripts\python.exe v13-test\verify_icp_job.py        # ICP-job variant comparison
```

Coverage: synthetic fit with known truth (RMS/scale/timing), partial fitting (4/12/24 markers), adaptive λ and LOO cache, reflection detection, anti-rigidity regression on real data (`ken5.csv`: ‖β‖ ≥ 15 at 24 markers), demographic prior (weakly constrained components → μ; backward-compat without prior), fresh cranio reimport from stale `sys.modules`, clean unregister/re-register (F8); V13.2: the dense set (scalp/regions), ICP on a synthetic skull with known transform (scale/rotation/translation recovered), full ICP+deform job (mesh wraps the skull at ~6 mm), dense-only fit (0 markers), rejection on a partial skull (chin excluded, scalp kept); V13.3–V13.4: nasal bridge patch on the bony bridge (regression), Rhinion=12310 (iBUG 29), huber_rows IRLS scoping; V13.6: landmark registration/mirror constraints for Acanthion + Piriform, precedence chain (manual > V12 > JSON) on the new landmarks, synthetic tangent/intersection cases with known angle (like the Umeyama/TPS tests), the isotropy guard, the offline report block, and the live cyan ghost (worker → timer application, hide on invalid estimate).

The harnesses measure, on the real ken-13test case: nasal bridge distance to bone (median and _signed_ — positive = skin above bone, negative = flattened), bridge keep-rate, skull exposed through skin (excluding natural openings), per-band coverage, and β saturation (clip count / face displacement from the statistical mean).

## Known limitations

- **~1 cm residual in the live preview** — by design: the TPS correction exists only offline (scipy is missing from Blender). For publication, run the offline pipeline.
- **The Gerasimov nose-projection diagnostic (V13.6) has case-level accuracy of ~1.5–2 cm** (validated on ken-13test, both surfaces, with a consistent posterior-inferior sign) — it is an informational cross-check, _not_ a fit constraint. It is also sensitive to the exact Acanthion/Piriform placement (two near-parallel lines intersect far away) and to the skull's nasal-bone curvature near Rhinion (the 2 mm tangent patch can be near-isotropic → guarded fallback to the Nasion→Rhinion direction).
- **The ICP (V13.2) is a coarse stage**: it assumes a roughly upright skull (multi-start on yaw only) and has a residual error of a few degrees/~2 cm on the near-spherical vault (weak yaw constraint); final precision comes from the β fit + markers. With ≥3 markers placed, the marker fit is a better initialization than the multi-start.
- **With "Continuous Dense" active, the rate drops to ~2-3 Hz** (from ~10 Hz) — the fit is ~10× larger; the UI stays responsive (worker thread).
- **The demographic prior requires offline precomputation** (TensorFlow) — the `priors/*.npz` files are generated once with `make_demographic_prior.py`.
- **Hard ±3σ clip:** robust medieval skulls can push dozens of components to the limit (visible in the status line: `clip N`). This is the model's validity domain; the excess is handled by the offline TPS. An adjustable _Clip Sigma_ slider (Advanced, 2.5–4.0) exists, with a plausibility warning.
- **LOO-CV can hit the grid edge (0.3)** on atypical skulls — flagged as a warning (offline: `stability_warnings`).
- **GNM template micro-asymmetry** (<0.2 mm on some bilateral pairs) — a property of the v3.0 model, not a placement error.
- **"REFLECTION" fit error** = Left/Right markers swapped or grossly wrong placement (intentional safety behavior).
- **Per-area local-view** can be accidentally exited with Numpad-/ — the "Setup / Repair Layout" button rebuilds the layout.
- **pytest: 46/47** — one known, unrelated failure: `test_bilateral_landmark_vertices_are_exact_mirrors` (0.05 mm tolerance vs the template's 0.075 mm micro-asymmetry at Supraorbitale).

## References

- Google GNM (Generative Neural Model), Head v3.0 variant — `github.com/google/GNM`.
- Umeyama, S. (1991). _Least-squares estimation of transformation parameters between two point patterns._ IEEE PAMI 13(4).
- Bookstein, F. L. (1989). _Principal warps: Thin-plate splines and the decomposition of deformations._ IEEE PAMI 11(6).
- Rhine, J. S. & Campbell, H. R. (1980). _Thickness of facial tissues in American Blacks._ J. Forensic Sci. 25(4).
- De Greef, S. et al. (2006). _Large-scale in-vivo Caucasian facial soft tissue thickness database for craniofacial reconstruction._ Forensic Sci. Int. 159S.
- Stephan, C. N. & Simpson, E. K. (2008). _Facial soft tissue depths in craniofacial identification_ (review). J. Forensic Sci. 53(4).

## License

**Apache License 2.0** (inherited from the Google GNM repo). The GNM Head v3.0 model and official landmarks belong to the Google GNM project; see the terms in `LICENSE` and `CONTRIBUTING.md`.

---

Built with ❤️ by Claude and Kimi K3
_GNM Craniofacial Approximation · Blender addon V13.5 + offline pipeline (cranio v4.0) · 



