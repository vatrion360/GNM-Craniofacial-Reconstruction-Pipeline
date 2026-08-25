# -*- coding: utf-8 -*-
"""Utilitare de geometrie: normale, masti de regiuni, esantionare cranii,
corespondente dense model->craniu.

Nu stie nimic despre optimizare; primeste si returneaza arrays/dict-uri.
"""

import numpy as np

from .landmarks import (NOSE_BRIDGE_CENTER_FRAC, NOSE_BRIDGE_RADIUS_MM,
                        NOSE_BRIDGE_TIP_CUT_MM, PROTECTED_GROUPS,
                        REGION_OFFSETS_MM)


def compute_vertex_normals(vertices, triangles, flip=False):
    """Normale per-vertex (acumulare produse vectoriale ale fetelor)."""
    tri = triangles[:, ::-1] if flip else triangles
    v0 = vertices[tri[:, 0]]
    face_n = np.cross(vertices[tri[:, 1]] - v0, vertices[tri[:, 2]] - v0)
    vn = np.zeros_like(vertices)
    np.add.at(vn, tri[:, 0], face_n)
    np.add.at(vn, tri[:, 1], face_n)
    np.add.at(vn, tri[:, 2], face_n)
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(n, 1e-12)


def build_scalp_mask(mu, vertex_groups, vertex_group_names):
    """Masca de scalp: vertecsii GNM a caror piele acopera direct osul.

    = skin AND NOT ears AND (NOT hockey_mask OR frunte OR tample)
      AND deasupra unui plan inferior inclinat (anterior: sprancene,
      posterior: baza occiputului), care exclude gatul.
    Validata pe template-ul GNM v3.0 (~2280 vertecsi pe calvarie; include
    Vertex si Eurion, exclude fata/urechile/gatul).
    """
    def grp(name):
        return vertex_groups[vertex_group_names.index(name)] > 0.5

    skin = grp("skin")
    ears = grp("ears")
    hockey = grp("hockey_mask")
    forehead = grp("forehead_region")
    temples = grp("left_temple_region") | grp("right_temple_region")

    v_m = mu / 1000.0  # pragurile sunt derivate in metri, pe template
    lower_plane = 0.31 + 0.238 * (v_m[:, 2] - 0.12)
    mask = skin & ~ears & (~hockey | forehead | temples) & (
        v_m[:, 1] > lower_plane)
    return np.where(mask)[0]


def build_face_dense_regions(mu, vertex_groups, vertex_group_names,
                             pronasale_vid=12296, nasion_vid=12319):
    """Regiuni faciale cu tesut subtire, pentru constrangeri dense.

    Returneaza lista de (nume_regiune, idx_vertecsi, offset_mm). Excluse
    deliberate (tesut gros/mobil sau fara os sub piele): varful nazal si
    alarele, buzele (s-ar atrage de dinti!), obrajii, parotidele, orbitele.

    Puntea nazala (V13.3): patch-ul este centrat pe PUNTEA OSOASA -
    centrul = Nasion + NOSE_BRIDGE_CENTER_FRAC * (pronasale - Nasion),
    cu tot ce e sub pronasale + NOSE_BRIDGE_TIP_CUT_MM exclus. Referinta
    este PRONASALE (vertex 12296, varful nasului); pana in V13.3 patch-ul
    era centrat direct pe acest vertex (sub numele eronat "Rhinion"), deci
    constrangea varful/apertura (fara os; corespondentele cadeau pe
    marginile aperturii sau erau respinse, iar puntea ramanea turtita).
    """
    def grp(name):
        return vertex_groups[vertex_group_names.index(name)] > 0.5

    skin = grp("skin")
    bridge_center = (mu[nasion_vid] + NOSE_BRIDGE_CENTER_FRAC
                     * (mu[pronasale_vid] - mu[nasion_vid]))
    nose_bridge = (grp("nose_region")
                   & (np.linalg.norm(mu - bridge_center, axis=1)
                      < NOSE_BRIDGE_RADIUS_MM)
                   & (mu[:, 1] > mu[pronasale_vid, 1] + NOSE_BRIDGE_TIP_CUT_MM))

    regions = [
        ("punte_nazala", nose_bridge, REGION_OFFSETS_MM["punte_nazala"]),
        ("brow_median", grp("middle_brow_region"),
         REGION_OFFSETS_MM["brow_median"]),
        ("zigomatic", grp("left_zygomatic_region")
         | grp("right_zygomatic_region"), REGION_OFFSETS_MM["zigomatic"]),
        ("infraorbital", grp("left_infraorbital_region")
         | grp("right_infraorbital_region"), REGION_OFFSETS_MM["infraorbital"]),
        ("barbie", grp("chin_region"), REGION_OFFSETS_MM["barbie"]),
    ]
    return [(name, np.where(skin & mask)[0], off)
            for name, mask, off in regions]


def build_vertex_region_map(num_vertices, scalp_idx, face_regions):
    """vertex -> nume regiune densa (sau None) - pentru raportul per regiune."""
    region_of = np.full(num_vertices, "", dtype=object)
    region_of[scalp_idx] = "scalp"
    for name, idx, _off in face_regions:
        region_of[idx] = name
    return region_of


def build_protected_mask(vertex_groups, vertex_group_names):
    """Indicii vertecsilor din grupurile fara ancore anatomice (ochi/buze/
    interior gura) - deplasarea lor TPS se amortizeaza (protect-damping)."""
    mask = np.zeros(vertex_groups.shape[1], dtype=bool)
    for name in PROTECTED_GROUPS:
        if name in vertex_group_names:
            mask |= vertex_groups[vertex_group_names.index(name)] > 0.5
    return np.where(mask)[0]


def load_skull_samples(skull_path, n_samples=200000, seed=42):
    """Incarca craniul si esantioneaza puncte + normale pe suprafata lui.

    Se asteapta coordonate in milimetri, in acelasi spatiu world Blender ca
    CSV-ul markerilor (export STL din scena: fara conversii de axe).
    """
    import trimesh

    mesh = trimesh.load(skull_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError(f"Nu am putut citi un mesh valid din {skull_path}")
    points, face_idx = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
    normals = mesh.face_normals[face_idx]
    return mesh, np.asarray(points), np.asarray(normals)


def dense_correspondences(v_world, dense):
    """Corespondente vertecsi-constransi->craniu, cu respingerea celor invalide.

    Pentru fiecare vertex din dense["dense_idx"] (scalp + regiuni faciale):
    cel mai apropiat punct esantionat pe craniu. Se resping corespondentele
    cu distanta > max_dist_mm (craniu partial / zona lipsa, ex. mandibula
    lipsa) sau cu normala craniului aproape opusa celei a modelului (tablita
    interna, muchii taiate). Tintele = punct + offset_regiune * normala.
    Returneaza (idx_pastrati, tinte_world, masca_keep, distanta_medie).
    """
    dense_idx = dense["dense_idx"]
    pts = v_world[dense_idx]
    dists, nn = dense["tree"].query(pts)
    closest = dense["points"][nn]
    n_skull = dense["normals"][nn]
    n_model = compute_vertex_normals(v_world, dense["triangles"],
                                     flip=dense.get("flip", False))[dense_idx]
    dots = np.einsum("ij,ij->i", n_model, n_skull)

    # La primul apel: detecteaza daca orientarea fetelor GNM e inversa fata
    # de conventia craniului si corecteaza global (o singura data).
    if dense.get("flip") is None:
        dense["flip"] = bool(np.median(dots) < 0)
        if dense["flip"]:
            return dense_correspondences(v_world, dense)

    keep = (dists < dense["max_dists"]) & (dots > dense["min_dot"])
    sidx = dense_idx[keep]
    targets_w = closest[keep] + dense["offsets"][keep, None] * n_skull[keep]
    mean_dist = float(dists[keep].mean()) if keep.any() else float("nan")
    return sidx, targets_w, keep, mean_dist
