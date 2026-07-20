"""
Pipeline Reconstructie Craniofaciala GNM - V7
Sistem Hibrid (Vertex Exact + Baricentric). Executie stricta din consola.
"""

import argparse
import csv
import numpy as np
import trimesh
from gnm.shape import gnm_numpy
from gnm.shape import gnm_landmarks

def _decode_exact_index(gnm_index: int) -> int:
    return -gnm_index - 1

def _get_template_position_and_basis(gnm, lm, index: int):
    # Dacă indexul este pozitiv -> Punct Baricentric (ex: Nasospinale)
    if index >= 0:
        w = lm.weights[index]
        idxs = lm.indices[index]
        pos = np.einsum("ji,j->i", gnm.template_vertex_positions[idxs], w)
        basis = np.einsum("kji,j->ki", gnm.vertex_identity_basis[:, idxs], w)
        return pos, basis
    
    # Dacă indexul este negativ -> Punct Exact (ex: Nasion, Gnathion)
    v_id = _decode_exact_index(index)
    return gnm.template_vertex_positions[v_id], gnm.vertex_identity_basis[:, v_id]

def load_target_points(csv_path: str) -> dict[int, np.ndarray]:
    targets = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["gnm_landmark_index"])
            point = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
            if np.array_equal(point, np.zeros(3)):
                continue
            targets[idx] = point
    return targets

def umeyama_alignment(source: np.ndarray, target: np.ndarray):
    mu_src, mu_tgt = source.mean(0), target.mean(0)
    src_c, tgt_c = source - mu_src, target - mu_tgt
    cov = tgt_c.T @ src_c / len(source)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        print("\n[EROARE FATALA] Alinierea a esuat (Reflexie Stanga/Dreapta).")
        print("Markerii au fost inversati in ciuda protectiei. Verifica CSV-ul.")
        S[-1, -1] = -1
        
    R = U @ S @ Vt
    var_src = (src_c ** 2).sum() / len(source)
    s = np.trace(np.diag(D) @ S) / var_src
    t = mu_tgt - s * R @ mu_src
    return R, s, t

def run_fitting(gnm: gnm_numpy.GNM, target_points: dict[int, np.ndarray], reg: float):
    lm = gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68)
    subset = sorted(target_points.keys())
    
    positions_and_bases = [_get_template_position_and_basis(gnm, lm, i) for i in subset]
    template_subset = np.stack([p for p, _ in positions_and_bases])
    basis_subset = np.stack([b for _, b in positions_and_bases], axis=1)
    target_raw = np.stack([target_points[i] for i in subset])

    R, s, t = umeyama_alignment(target_raw, template_subset)
    target_aligned = (s * (R @ target_raw.T).T) + t

    pre_fit_error = np.linalg.norm(target_aligned - template_subset, axis=-1)
    print(f"Eroare medie post-aliniere spatiala (Umeyama): {pre_fit_error.mean()*1000:.2f} mm")

    A = basis_subset.reshape(gnm.identity_dim, -1).T
    b = (target_aligned - template_subset).reshape(-1)
    n = A.shape[0]
    
    identity_params = A.T @ np.linalg.solve(A @ A.T + reg * np.eye(n), b)

    vertices = gnm(
        identity=identity_params,
        expression=np.zeros(gnm.expression_dim),
        rotations=np.zeros((gnm.num_joints, 3)),
        translation=np.zeros(3)
    )

    recon = template_subset + np.einsum("k,knw->nw", identity_params, basis_subset)
    post_fit_error = np.linalg.norm(recon - target_aligned, axis=-1)
    print(f"Eroare medie dupa deformarea fetei (Ridge): {post_fit_error.mean()*1000:.2f} mm")

    return vertices, (R, s, t)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Fisier CSV din Blender")
    parser.add_argument("--output", default="reconstructie_v7.obj")
    parser.add_argument("--regularization", type=float, default=1e-5) 
    args = parser.parse_args()

    print("Incarcare model GNM (Variant HEAD)...")
    gnm = gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    
    targets = load_target_points(args.input)
    print(f"S-au incarcat {len(targets)} markeri valizi.")

    vertices, transform = run_fitting(gnm, targets, args.regularization)

    print("Exportare .obj in spatiul metric Blender...")
    R, s, t = transform
    R_inv = np.linalg.inv(R)
    aligned_vertices = ((vertices - t) @ R_inv.T) / s
    
    mesh = trimesh.Trimesh(vertices=aligned_vertices, faces=gnm.triangles, process=False)
    mesh.export(args.output, file_type="obj")
    print(f"Finalizat: {args.output}")

if __name__ == "__main__":
    main()