import copy
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment

from OCC.Core.gp import gp_Pnt
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface

from cut_faces.utils_occ import get_edges_from_face
from cut_faces.utils_fit import (
    fit_cone,
    fit_cylinder,
    fit_sphere,
    fit_torus,
)


TWO_PI = 2.0 * np.pi

# -----------------------------
# Basic interval/math utilities
# -----------------------------
def wrap_to_pi(x: float) -> float:
    return (float(x) + np.pi) % TWO_PI - np.pi


def normalize_periodic_interval(v_min: float, v_max: float, period: float = TWO_PI) -> Tuple[float, float, float]:
    span = (float(v_max) - float(v_min)) % period
    if span < 1e-9:
        span = period
    start = float(v_min) % period
    return start, start + span, span


def normalize_u_interval(u_min: float, u_max: float) -> Tuple[float, float, float]:
    _, _, span = normalize_periodic_interval(u_min, u_max, period=TWO_PI)
    start = wrap_to_pi(u_min)
    return start, start + span, span


def normalize_v_interval(face_type: str, v_min: float, v_max: float) -> Tuple[float, float, float, bool]:
    if face_type == "torus":  # sphere is 0-pi and not period
        s, e, span = normalize_periodic_interval(v_min, v_max, period=TWO_PI)
        return s, e, span, True
    s, e = float(v_min), float(v_max)
    return s, e, e - s, False


def periodic_order(starts_mod: List[float], period: float = TWO_PI) -> List[int]:
    n = len(starts_mod)
    if n <= 1:
        return list(range(n))
    order = np.argsort(np.asarray(starts_mod, dtype=float))
    vals = np.asarray([starts_mod[i] for i in order], dtype=float)
    gaps = np.diff(np.r_[vals, vals[0] + period])
    split = int(np.argmax(gaps))
    return list(np.r_[order[split + 1 :], order[: split + 1]])


def circular_mean(values: List[float], period: float = TWO_PI) -> float:
    vals = np.asarray(values, dtype=float) * (TWO_PI / period)
    return float(np.arctan2(np.sin(vals).sum(), np.cos(vals).sum()) % TWO_PI) * (period / TWO_PI)


def circular_distance(a: float, b: float, period: float = TWO_PI) -> float:
    d = (float(a) - float(b) + 0.5 * period) % period - 0.5 * period
    return abs(d)


# -----------------------------
# Quantization helpers
# -----------------------------
def estimate_cycle_count(spans: List[float], period: float = TWO_PI, tol_ratio: float = 0.2) -> int:
    total = float(np.sum(spans))
    k = int(np.round(total / period))
    k = max(1, min(k, len(spans)))
    if abs(total - k * period) > tol_ratio * period:
        return 0
    return k


def periodic_group_labels(starts: List[float], k: int, period: float = TWO_PI) -> List[int]:
    n = len(starts)
    if n == 0 or k <= 1:
        return [0] * n
    k = min(k, n)
    order = periodic_order([float(s % period) for s in starts], period=period)
    labels = [0] * n
    for rank, idx in enumerate(order):
        labels[idx] = min(k - 1, int(np.floor(rank * k / n)))
    return labels


def quantize_partition_spans(spans: List[float], period: float, max_den: int = 24) -> np.ndarray:
    arr = np.asarray(spans, dtype=float)
    n = arr.size
    if n == 0:
        return arr

    total = float(arr.sum())
    if total < 1e-12:
        return np.full(n, period / n, dtype=float)

    fracs = arr / total
    qmin = max(1, n)
    best_err = np.inf
    best = None
    best_q = qmin

    for q in range(qmin, max(qmin, max_den) + 1):
        raw = fracs * q
        cnt = np.rint(raw).astype(int)
        cnt = np.maximum(cnt, 1)
        diff = int(q - cnt.sum())
        residual = raw - cnt

        if diff > 0:
            for i in np.argsort(-residual)[:diff]:
                cnt[i] += 1
        elif diff < 0:
            need = -diff
            for i in np.argsort(residual):
                if need <= 0:
                    break
                if cnt[i] > 1:
                    take = min(need, cnt[i] - 1)
                    cnt[i] -= take
                    need -= take
            if need > 0:
                continue

        approx = cnt / q
        err = float(np.abs(approx - fracs).sum()) + 1e-6 * q
        if err < best_err:
            best_err = err
            best = cnt.copy()
            best_q = q

    if best is None:
        return arr / (arr.sum() + 1e-12) * period
    return best.astype(float) * (period / best_q)


def quantized_intervals(
    keys: List[int],
    starts: List[float],
    spans: List[float],
    period: float,
    max_den: int = 24,
    periodic: bool = True,
) -> Dict[int, Tuple[float, float]]:
    order = periodic_order([float(s % period) for s in starts], period=period) if periodic else list(np.argsort(np.asarray(starts, dtype=float)))
    q_spans = quantize_partition_spans([spans[i] for i in order], period=period, max_den=max_den)

    out: Dict[int, Tuple[float, float]] = {}
    cursor = starts[order[0]] % period if periodic else 0.0
    for i, span in zip(order, q_spans):
        span = float(max(1e-8, span))
        out[keys[i]] = (cursor, cursor + span)
        cursor += span
    return out


def label_interval_stats(
    starts: List[float],
    spans: List[float],
    labels: List[int],
    k: int,
    period: float,
    periodic: bool,
) -> Tuple[List[int], List[float], List[float]] | None:
    keys, centers, widths = [], [], []
    for g in range(k):
        ids = [i for i, lb in enumerate(labels) if lb == g]
        if not ids:
            return None
        keys.append(g)
        if periodic:
            centers.append(circular_mean([starts[i] % period for i in ids], period=period))
        else:
            centers.append(float(np.mean([starts[i] for i in ids])))
        widths.append(float(np.mean([spans[i] for i in ids])))
    return keys, centers, widths


def assign_unique_uv_cells(
    u_starts: List[float],
    v_starts: List[float],
    u_labels: List[int],
    v_labels: List[int],
    u_k: int,
    v_k: int,
    v_period: float = TWO_PI,
    v_periodic: bool = True,
) -> Tuple[List[int], List[int]]:
    n = len(u_starts)
    if n == 0 or u_k <= 1 or v_k <= 1 or n != u_k * v_k:
        return u_labels, v_labels
    if len({(u_labels[i], v_labels[i]) for i in range(n)}) == n:
        return u_labels, v_labels

    def centers(starts: List[float], labels: List[int], k: int, period: float, periodic: bool) -> List[float]:
        base = circular_mean([s % period for s in starts], period=period) if periodic else float(np.mean(starts))
        out = []
        for g in range(k):
            ids = [i for i, lb in enumerate(labels) if lb == g]
            if not ids:
                step = period * g / max(k, 1)
                out.append((base + step) % period if periodic else base + step)
            else:
                if periodic:
                    out.append(circular_mean([starts[i] % period for i in ids], period=period))
                else:
                    out.append(float(np.mean([starts[i] for i in ids])))
        return out

    uc = centers(u_starts, u_labels, u_k, TWO_PI, True)
    vc = centers(v_starts, v_labels, v_k, v_period, v_periodic)

    cells = [(iu, iv) for iu in range(u_k) for iv in range(v_k)]
    cost = np.zeros((n, n), dtype=float)
    for i in range(n):
        for c, (iu, iv) in enumerate(cells):
            v_cost = circular_distance(v_starts[i], vc[iv], period=v_period) if v_periodic else abs(v_starts[i] - vc[iv])
            cost[i, c] = circular_distance(u_starts[i], uc[iu]) + v_cost

    row_ind, col_ind = linear_sum_assignment(cost)
    if len(row_ind) != n:
        return u_labels, v_labels

    out_u = [0] * n
    out_v = [0] * n
    for i, c in zip(row_ind, col_ind):
        out_u[int(i)], out_v[int(i)] = cells[int(c)]
    return out_u, out_v


# -----------------------------
# Geometry helpers
# -----------------------------
def canonicalize_axis(axis: np.ndarray) -> np.ndarray:
    a = np.asarray(axis, dtype=float).reshape(3)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    a = a / n
    if a[int(np.argmax(np.abs(a)))] < 0:
        a = -a
    return a


def build_frame_from_axis(axis: np.ndarray, hints: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = canonicalize_axis(axis)
    u_sum = np.zeros(3, dtype=float)

    for h in hints:
        hv = np.asarray(h, dtype=float).reshape(3)
        hv = hv - np.dot(hv, axis) * axis
        n = np.linalg.norm(hv)
        if n < 1e-12:
            continue
        hv = hv / n
        if np.dot(hv, u_sum) < 0:
            hv = -hv
        u_sum += hv

    if np.linalg.norm(u_sum) < 1e-12:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(ref, axis)) > 0.95:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        u = ref - np.dot(ref, axis) * axis
        u = u / (np.linalg.norm(u) + 1e-12)
    else:
        u = u_sum / (np.linalg.norm(u_sum) + 1e-12)

    v = np.cross(axis, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return axis, u, v


def is_axis_close(a: np.ndarray, b: np.ndarray, deg: float = 10.0) -> bool:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    ang = np.arccos(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return ang < np.deg2rad(deg)


def is_centers_coaxial(c1: np.ndarray, c2: np.ndarray, axis: np.ndarray, tol: float = 0.12) -> bool:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    d = c2 - c1
    off = d - np.dot(d, axis) * axis
    return np.linalg.norm(off) < tol


# -----------------------------
# Fit / resample
# -----------------------------
FITTERS = {
    "cylinder": fit_cylinder,
    "torus": fit_torus,
    "sphere": fit_sphere,
    "cone": fit_cone,
}


def fit_face(face_type: str, points: np.ndarray) -> Dict:
    if face_type not in FITTERS:
        raise ValueError(f"Unsupported type: {face_type}")
    fit_pts, err, ok, params = FITTERS[face_type](points)
    if fit_pts is None or params is None:
        raise RuntimeError(f"fit failed for {face_type}")
    return {
        "type": face_type,
        "orig_points": points,
        "fit_points": fit_pts,
        "params": params,
        "ok": bool(ok),
        "err": float(err),
    }


def fit_best_face(points: np.ndarray) -> Dict | None:
    fit_data = {}
    for face_type, fitter in FITTERS.items():
        fit_pts, err, ok, params = fitter(points)
        if ok and fit_pts is not None and params is not None:
            fit_data[face_type] = (float(err), fit_pts, params)

    if not fit_data:
        return None

    face_type, (err, fit_pts, params) = min(fit_data.items(), key=lambda kv: kv[1][0])
    if err > 0.025:  # fit error gate
        return None

    return {
        "type": face_type,
        "orig_points": points,
        "fit_points": fit_pts,
        "params": params,
        "ok": True,
        "err": err,
    }


def write_interval(
    params: Dict,
    mode: str,
    start: float,
    end: float,
    low: float = 0.0,
    high: float = np.pi,
) -> None:
    span = float(end - start)
    if mode == "u_periodic":
        key = "u"
        s = wrap_to_pi(start)
    elif mode == "v_periodic":
        key = "v"
        s = float(start) % TWO_PI
    elif mode == "v_linear":
        key = "v"
        span = max(1e-6, span)
        max_start = high - span
        if max_start < low:
            span = max(1e-6, high - low)
            max_start = low
        s = min(max(float(start), low), max_start)
    else:
        raise ValueError(f"Unsupported interval mode: {mode}")
    params[f"{key}_min"] = s
    params[f"{key}_max"] = s + span


def refresh_face(face: Dict) -> None:
    n_v, n_u, _ = face["fit_points"].shape
    face_type = face["type"]
    p = face["params"]
    axis = np.asarray(p["axis_dir"], dtype=float)
    u_dir = np.asarray(p["u_dir"], dtype=float)
    v_dir = np.asarray(p["v_dir"], dtype=float)
    center = np.asarray(p["center"], dtype=float)

    if face_type == "cylinder":
        radius = float(p["radius"])
        v_min, v_max = float(p["v_min"]), float(p["v_max"])
        h = float(p.get("height", v_max - v_min))
        U, Vh = np.meshgrid(np.linspace(float(p["u_min"]), float(p["u_max"]), n_u), np.linspace(v_min, v_max, n_v))
        face["fit_points"] = (
            center[None, None, :]
            + radius * np.cos(U)[:, :, None] * u_dir[None, None, :]
            + radius * np.sin(U)[:, :, None] * v_dir[None, None, :]
            + (Vh - (v_min + 0.5 * h))[:, :, None] * axis[None, None, :]
        )
    elif face_type == "torus":
        R, r = float(p["radius"]), float(p["r"])
        U, V = np.meshgrid(
            np.linspace(float(p["u_min"]), float(p["u_max"]), n_u),
            np.linspace(float(p["v_min"]), float(p["v_max"]), n_v),
        )
        X = (R + r * np.cos(V)) * np.cos(U)
        Y = (R + r * np.cos(V)) * np.sin(U)
        Z = r * np.sin(V)
        local = np.stack([X, Y, Z], axis=2).reshape(-1, 3)
        Rm = np.stack([u_dir, v_dir, axis], axis=1)
        face["fit_points"] = ((Rm @ local.T).T + center).reshape(n_v, n_u, 3)
    elif face_type == "sphere":
        radius = float(p["radius"])
        U, V = np.meshgrid(
            np.linspace(float(p["u_min"]), float(p["u_max"]), n_u),
            np.linspace(float(p["v_min"]), float(p["v_max"]), n_v),
        )
        X = radius * np.sin(V) * np.cos(U)
        Y = radius * np.sin(V) * np.sin(U)
        Z = radius * np.cos(V)
        local = np.stack([X, Y, Z], axis=2).reshape(-1, 3)
        Rm = np.stack([u_dir, v_dir, axis], axis=1)
        face["fit_points"] = ((Rm @ local.T).T + center).reshape(n_v, n_u, 3)
    elif face_type == "cone":
        k, c = float(p["k"]), float(p["c"])
        U, S = np.meshgrid(
            np.linspace(float(p["u_min"]), float(p["u_max"]), n_u),
            np.linspace(float(p["v_min"]), float(p["v_max"]), n_v),
        )
        R = np.maximum(k * S + c, 1e-8)
        face["fit_points"] = (
            center[None, None, :]
            + S[:, :, None] * axis[None, None, :]
            + R[:, :, None] * (np.cos(U)[:, :, None] * u_dir[None, None, :] + np.sin(U)[:, :, None] * v_dir[None, None, :])
        )
    else:
        raise ValueError(f"Unsupported type: {face_type}")

    p["e"] = [
        face["fit_points"][0],
        face["fit_points"][-1],
        face["fit_points"][:, 0],
        face["fit_points"][:, -1],
    ]


# -----------------------------
# Complementary matching / alignment
# -----------------------------
def best_interval_pair(p1: Dict, p2: Dict) -> Dict:
    s1, e1, l1 = normalize_u_interval(p1["u_min"], p1["u_max"])
    s2, e2, l2 = normalize_u_interval(p2["u_min"], p2["u_max"])
    best = None

    for k in (-1, 0, 1):
        s2k, e2k = s2 + k * TWO_PI, e2 + k * TWO_PI
        if s1 <= s2k:
            first, second = ("a", s1, e1, l1), ("b", s2k, e2k, l2)
        else:
            first, second = ("b", s2k, e2k, l2), ("a", s1, e1, l1)

        gap = max(0.0, second[1] - first[2])
        overlap = max(0.0, first[2] - second[1])
        cover = max(first[2], second[2]) - min(first[1], second[1])
        score = gap + overlap + abs(cover - TWO_PI)

        cand = {
            "first": first,
            "second": second,
            "l1": l1,
            "l2": l2,
            "gap": gap,
            "overlap": overlap,
            "score": score,
        }
        if best is None or cand["score"] < best["score"]:
            best = cand

    return best


def is_sphere_v_complement_mode(pa: Dict, pb: Dict) -> Tuple[bool, Dict]:
    u1 = normalize_u_interval(pa["u_min"], pa["u_max"])[2]
    u2 = normalize_u_interval(pb["u_min"], pb["u_max"])[2]
    u_full = (u1 > 5.2) and (u2 > 5.2)  # sphere u coverage

    s1, e1, l1, _ = normalize_v_interval("sphere", pa["v_min"], pa["v_max"])
    s2, e2, l2, _ = normalize_v_interval("sphere", pb["v_min"], pb["v_max"])
    if s1 <= s2:
        first, second = ("a", s1, e1, l1), ("b", s2, e2, l2)
    else:
        first, second = ("b", s2, e2, l2), ("a", s1, e1, l1)

    seam = max(0.0, second[1] - first[2]) + max(0.0, first[2] - second[1])
    v_comp = abs((l1 + l2) - np.pi) < 0.45 and seam < 0.30  # sphere v match
    return bool(u_full and v_comp), {"first": first, "second": second}


def is_radius_close(face_type: str, pa: Dict, pb: Dict) -> bool:
    if face_type == "cylinder":
        return abs(pa["radius"] - pb["radius"]) <= 0.16  # cylinder radius
    if face_type == "torus":
        return abs(pa["radius"] - pb["radius"]) <= 0.16 and abs(pa["r"] - pb["r"]) <= 0.10  # torus radii
    if face_type == "sphere":
        return abs(pa["radius"] - pb["radius"]) <= 0.16  # sphere radius
    if face_type == "cone":
        return abs(pa["radius_max"] - pb["radius_max"]) <= 0.18  # cone radius
    return False


def is_complementary_pair(face_a: Dict, face_b: Dict, require_interval: bool = True) -> bool:
    if face_a["type"] != face_b["type"] or face_a["type"] not in {"cylinder", "torus", "sphere", "cone"}:
        return False

    pa, pb = face_a["params"], face_b["params"]
    if not is_axis_close(pa["axis_dir"], pb["axis_dir"], deg=10.0):  # axis direction
        return False

    axis = canonicalize_axis(np.asarray(pa["axis_dir"], dtype=float) + np.asarray(pb["axis_dir"], dtype=float))
    if not is_centers_coaxial(pa["center"], pb["center"], axis, tol=0.12):  # coaxial center
        return False

    face_type = face_a["type"]
    if not is_radius_close(face_type, pa, pb):
        return False

    if not require_interval:
        return True

    if face_type == "sphere" and is_sphere_v_complement_mode(pa, pb)[0]:
        return True

    iv = best_interval_pair(pa, pb)
    return abs((iv["l1"] + iv["l2"]) - TWO_PI) < 0.45 and (iv["gap"] + iv["overlap"]) < 0.55  # interval complement


def align_complementary_faces(faces: List[Dict]) -> List[Dict]:
    """
    Align one pre-selected complementary face group.
    """
    aligned_faces = copy.deepcopy(faces)
    if len(aligned_faces) < 2:
        return aligned_faces

    face_count = len(aligned_faces)
    if face_count not in {2, 4}:
        return aligned_faces

    if face_count == 2 and not is_complementary_pair(aligned_faces[0], aligned_faces[1]):
        return aligned_faces

    face_type = aligned_faces[0]["type"]
    if face_type not in {"cylinder", "torus", "sphere", "cone"}:
        return aligned_faces

    if any(face["type"] != face_type for face in aligned_faces):
        return aligned_faces

    params_list = [face["params"] for face in aligned_faces]

    axes, centers, u_hints = [], [], []
    for p in params_list:
        a = canonicalize_axis(p["axis_dir"])
        if axes and np.dot(a, axes[0]) < 0:
            a = -a
        axes.append(a)
        centers.append(np.asarray(p["center"], dtype=float))
        u_hints.append(np.asarray(p["u_dir"], dtype=float))

    axis = canonicalize_axis(np.sum(np.stack(axes), axis=0))
    p0 = np.mean(np.stack(centers), axis=0)

    def proj(c: np.ndarray) -> np.ndarray:
        return p0 + np.dot(c - p0, axis) * axis

    center = np.mean(np.stack([proj(c) for c in centers]), axis=0)
    axis, u, v = build_frame_from_axis(axis, u_hints)

    if face_type == "cylinder":
        rr = float(np.mean([p["radius"] for p in params_list]))
        for p in params_list:
            p["radius"] = rr
    elif face_type == "torus":
        RR = float(np.mean([p["radius"] for p in params_list]))
        rr = float(np.mean([p["r"] for p in params_list]))
        for p in params_list:
            p["radius"] = RR
            p["r"] = rr
    elif face_type == "sphere":
        rr = float(np.mean([p["radius"] for p in params_list]))
        for p in params_list:
            p["radius"] = rr
    elif face_type == "cone":
        kk = float(np.mean([p["k"] for p in params_list]))
        cc = float(np.mean([p["c"] for p in params_list]))
        for p in params_list:
            p["k"] = kk
            p["c"] = cc

    for p in params_list:
        p["axis_dir"], p["u_dir"], p["v_dir"], p["center"] = axis, u, v, center

    chunks_u = []
    for i, face in enumerate(aligned_faces):
        s, _, sp = normalize_u_interval(face["params"]["u_min"], face["params"]["u_max"])
        chunks_u.append((i, s % TWO_PI, sp))

    if face_count == 2:
        face_a, face_b = aligned_faces
        pa, pb = face_a["params"], face_b["params"]

        if face_type == "sphere" and is_sphere_v_complement_mode(pa, pb)[0]:
            sv = is_sphere_v_complement_mode(pa, pb)[1]
            first, second = sv["first"], sv["second"]
            l1, l2 = quantize_partition_spans([first[3], second[3]], period=np.pi, max_den=24)

            if first[0] == "a":
                write_interval(pa, "v_linear", 0.0, float(l1))
                write_interval(pb, "v_linear", float(l1), float(l1 + l2))
            else:
                write_interval(pb, "v_linear", 0.0, float(l1))
                write_interval(pa, "v_linear", float(l1), float(l1 + l2))

            u_start = circular_mean(
                [
                    normalize_u_interval(pa["u_min"], pa["u_max"])[0] % TWO_PI,
                    normalize_u_interval(pb["u_min"], pb["u_max"])[0] % TWO_PI,
                ]
            )
            write_interval(pa, "u_periodic", u_start, u_start + TWO_PI)
            write_interval(pb, "u_periodic", u_start, u_start + TWO_PI)
        else:
            iv = best_interval_pair(pa, pb)
            first, second = iv["first"], iv["second"]
            l1, l2 = quantize_partition_spans([first[3], second[3]], period=TWO_PI, max_den=24)

            gs = min(first[1], second[1])
            f_s, f_e = gs, gs + float(l1)
            s_s, s_e = f_e, f_e + float(l2)

            if first[0] == "a":
                write_interval(pa, "u_periodic", f_s, f_e)
                write_interval(pb, "u_periodic", s_s, s_e)
            else:
                write_interval(pb, "u_periodic", f_s, f_e)
                write_interval(pa, "u_periodic", s_s, s_e)

        for face in aligned_faces:
            refresh_face(face)
        return aligned_faces

    used_group_uv = False
    if face_type in {"torus", "sphere"} and face_count == 4:
        chunks_v = []
        for i, face in enumerate(aligned_faces):
            p = face["params"]
            s, _, sp, _ = normalize_v_interval(face_type, p["v_min"], p["v_max"])
            chunks_v.append((i, s, sp))

        u_k = estimate_cycle_count([x[2] for x in chunks_u], period=TWO_PI, tol_ratio=0.26)
        v_period = TWO_PI if face_type == "torus" else np.pi
        v_k = estimate_cycle_count([x[2] for x in chunks_v], period=v_period, tol_ratio=0.26)

        if u_k > 1 and v_k > 1 and len(chunks_u) == u_k * v_k:
            u_starts = [x[1] for x in chunks_u]
            u_spans = [x[2] for x in chunks_u]
            v_starts = [x[1] for x in chunks_v]
            v_spans = [x[2] for x in chunks_v]
            v_periodic = face_type == "torus"

            u_labels = periodic_group_labels(u_starts, k=u_k, period=TWO_PI)
            if v_periodic:
                v_labels = periodic_group_labels(v_starts, k=v_k, period=v_period)
            else:
                order = list(np.argsort(np.asarray(v_starts, dtype=float)))
                v_labels = [0] * len(v_starts)
                for rank, idx in enumerate(order):
                    v_labels[idx] = min(v_k - 1, int(np.floor(rank * v_k / len(v_starts))))

            u_labels, v_labels = assign_unique_uv_cells(
                u_starts,
                v_starts,
                u_labels,
                v_labels,
                u_k,
                v_k,
                v_period=v_period,
                v_periodic=v_periodic,
            )

            u_stats = label_interval_stats(u_starts, u_spans, u_labels, u_k, period=TWO_PI, periodic=True)
            v_stats = label_interval_stats(v_starts, v_spans, v_labels, v_k, period=v_period, periodic=v_periodic)
            if u_stats is not None and v_stats is not None:
                u_map = quantized_intervals(*u_stats, period=TWO_PI, max_den=24, periodic=True)
                v_map = quantized_intervals(*v_stats, period=v_period, max_den=24, periodic=v_periodic)
                u_ranges = {idx: u_map[u_labels[t]] for t, (idx, _, _) in enumerate(chunks_u)}
                v_ranges = {idx: v_map[v_labels[t]] for t, (idx, _, _) in enumerate(chunks_v)}

                for idx, face in enumerate(aligned_faces):
                    us, ue = u_ranges[idx]
                    vs, ve = v_ranges[idx]
                    write_interval(face["params"], "u_periodic", us, ue)
                    if face_type == "torus":
                        write_interval(face["params"], "v_periodic", vs, ve)
                    else:
                        write_interval(face["params"], "v_linear", vs, ve, low=0.0, high=np.pi)
                used_group_uv = True

    if not used_group_uv:
        u_k = estimate_cycle_count([x[2] for x in chunks_u], period=TWO_PI, tol_ratio=0.26)
        if u_k <= 0:
            return aligned_faces

        u_starts = [x[1] for x in chunks_u]
        u_spans = [x[2] for x in chunks_u]
        u_labels = periodic_group_labels(u_starts, k=u_k, period=TWO_PI)
        u_stats = label_interval_stats(u_starts, u_spans, u_labels, u_k, period=TWO_PI, periodic=True)
        if u_stats is None:
            return aligned_faces

        u_ranges = quantized_intervals(*u_stats, period=TWO_PI, max_den=24, periodic=True)
        for t, (idx, _, _) in enumerate(chunks_u):
            us, ue = u_ranges[u_labels[t]]
            write_interval(aligned_faces[idx]["params"], "u_periodic", us, ue)

    for face in aligned_faces:
        refresh_face(face)

    return aligned_faces


def detect_aligned_edges(faces: List[Dict], adj: np.ndarray) -> List[List[int]]:
    """
    Return edge flags as [u-, u+, v-, v+] for each face.
    """
    n = len(faces)
    flags = [[0, 0, 0, 0] for _ in range(n)]
    if n == 0:
        return flags

    cache = []
    for face in faces:
        p = face["params"]
        u_s, u_e, _ = normalize_u_interval(p["u_min"], p["u_max"])
        v_s, v_e, _, v_periodic = normalize_v_interval(face["type"], p["v_min"], p["v_max"])
        cache.append({"u_s": u_s, "u_e": u_e, "v_s": v_s, "v_e": v_e, "v_periodic": v_periodic})

    eps = 1e-5

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] != 1:
                continue

            pa, pb = faces[i]["params"], faces[j]["params"]
            if faces[i]["type"] != faces[j]["type"]:
                continue
            same_geometry = True
            for key in ["axis_dir", "u_dir", "v_dir", "center"]:
                if key not in pa or key not in pb:
                    same_geometry = False
                    break
                if not np.allclose(np.asarray(pa[key], dtype=float), np.asarray(pb[key], dtype=float), atol=1e-8, rtol=0.0):
                    same_geometry = False
                    break
            if not same_geometry:
                continue

            face_type = faces[i]["type"]
            if face_type == "cylinder":
                same_geometry = abs(float(pa["radius"]) - float(pb["radius"])) <= 1e-8
            elif face_type == "torus":
                same_geometry = abs(float(pa["radius"]) - float(pb["radius"])) <= 1e-8 and abs(float(pa["r"]) - float(pb["r"])) <= 1e-8
            elif face_type == "sphere":
                same_geometry = abs(float(pa["radius"]) - float(pb["radius"])) <= 1e-8
            elif face_type == "cone":
                same_geometry = abs(float(pa["k"]) - float(pb["k"])) <= 1e-8 and abs(float(pa["c"]) - float(pb["c"])) <= 1e-8
            else:
                same_geometry = False
            if not same_geometry:
                continue


            pi, pj = cache[i], cache[j]
            ui_segments = [(0.0, TWO_PI)] if pi["u_e"] - pi["u_s"] >= TWO_PI - 1e-9 else []
            if not ui_segments:
                us = float(pi["u_s"]) % TWO_PI
                ue = us + (pi["u_e"] - pi["u_s"])
                ui_segments = [(us, ue)] if ue <= TWO_PI else [(us, TWO_PI), (0.0, ue - TWO_PI)]

            uj_segments = [(0.0, TWO_PI)] if pj["u_e"] - pj["u_s"] >= TWO_PI - 1e-9 else []
            if not uj_segments:
                us = float(pj["u_s"]) % TWO_PI
                ue = us + (pj["u_e"] - pj["u_s"])
                uj_segments = [(us, ue)] if ue <= TWO_PI else [(us, TWO_PI), (0.0, ue - TWO_PI)]

            u_overlap = any((min(a1, b1) - max(a0, b0)) > 1e-9 for a0, a1 in ui_segments for b0, b1 in uj_segments)

            if pi["v_periodic"] and pj["v_periodic"]:
                vi_segments = [(0.0, TWO_PI)] if pi["v_e"] - pi["v_s"] >= TWO_PI - 1e-9 else []
                if not vi_segments:
                    vs = float(pi["v_s"]) % TWO_PI
                    ve = vs + (pi["v_e"] - pi["v_s"])
                    vi_segments = [(vs, ve)] if ve <= TWO_PI else [(vs, TWO_PI), (0.0, ve - TWO_PI)]
                vj_segments = [(0.0, TWO_PI)] if pj["v_e"] - pj["v_s"] >= TWO_PI - 1e-9 else []
                if not vj_segments:
                    vs = float(pj["v_s"]) % TWO_PI
                    ve = vs + (pj["v_e"] - pj["v_s"])
                    vj_segments = [(vs, ve)] if ve <= TWO_PI else [(vs, TWO_PI), (0.0, ve - TWO_PI)]
                v_overlap = any((min(a1, b1) - max(a0, b0)) > 1e-9 for a0, a1 in vi_segments for b0, b1 in vj_segments)
            else:
                v_overlap = (min(pi["v_e"], pj["v_e"]) - max(pi["v_s"], pj["v_s"])) > 1e-9

            if v_overlap:
                if circular_distance(pi["u_s"], pj["u_e"], period=TWO_PI) <= eps:
                    flags[i][0] = 1
                    flags[j][1] = 1
                if circular_distance(pi["u_e"], pj["u_s"], period=TWO_PI) <= eps:
                    flags[i][1] = 1
                    flags[j][0] = 1
                if circular_distance(pi["u_s"], pj["u_s"], period=TWO_PI) <= eps:
                    flags[i][0] = 1
                    flags[j][0] = 1
                if circular_distance(pi["u_e"], pj["u_e"], period=TWO_PI) <= eps:
                    flags[i][1] = 1
                    flags[j][1] = 1

            if u_overlap:
                if pi["v_periodic"] and pj["v_periodic"]:
                    if circular_distance(pi["v_s"], pj["v_e"], period=TWO_PI) <= eps:
                        flags[i][2] = 1
                        flags[j][3] = 1
                    if circular_distance(pi["v_e"], pj["v_s"], period=TWO_PI) <= eps:
                        flags[i][3] = 1
                        flags[j][2] = 1
                    if circular_distance(pi["v_s"], pj["v_s"], period=TWO_PI) <= eps:
                        flags[i][2] = 1
                        flags[j][2] = 1
                    if circular_distance(pi["v_e"], pj["v_e"], period=TWO_PI) <= eps:
                        flags[i][3] = 1
                        flags[j][3] = 1
                else:
                    if abs(pi["v_s"] - pj["v_e"]) <= eps:
                        flags[i][2] = 1
                        flags[j][3] = 1
                    if abs(pi["v_e"] - pj["v_s"]) <= eps:
                        flags[i][3] = 1
                        flags[j][2] = 1
                    if abs(pi["v_s"] - pj["v_s"]) <= eps:
                        flags[i][2] = 1
                        flags[j][2] = 1
                    if abs(pi["v_e"] - pj["v_e"]) <= eps:
                        flags[i][3] = 1
                        flags[j][3] = 1

    pole_eps = 1e-5
    for i, face in enumerate(faces):
        if face["type"] != "sphere":
            continue
        p = face["params"]
        v_min, v_max, _, _ = normalize_v_interval(face["type"], p["v_min"], p["v_max"])
        if abs(v_min - 0.0) <= pole_eps:
            flags[i][2] = 1
        if abs(v_max - np.pi) <= pole_eps:
            flags[i][3] = 1

    return flags


# -----------------------------
# NPY pipeline
# -----------------------------
def fit_solid_faces(solid_surfs: np.ndarray, solid_mask: np.ndarray) -> Tuple[List[int], List[Dict]]:
    valid_indices: List[int] = []
    fitted_faces: List[Dict] = []
    for idx, points in enumerate(solid_surfs):
        if not bool(solid_mask[idx]):
            continue
        face = fit_best_face(points)
        if face is None:
            continue
        face["source_index"] = idx
        valid_indices.append(idx)
        fitted_faces.append(face)
    return valid_indices, fitted_faces


def compatible_components(compatible: np.ndarray, indices: List[int]) -> List[List[int]]:
    index_set = set(indices)
    seen = set()
    components: List[List[int]] = []

    for start in indices:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []

        while stack:
            i = stack.pop()
            component.append(i)
            for j in index_set:
                if j not in seen and compatible[i, j]:
                    seen.add(j)
                    stack.append(j)

        components.append(sorted(component))

    return components


def find_alignment_groups(faces: List[Dict], adj: np.ndarray) -> List[List[int]]:
    n = len(faces)
    if n == 0:
        return []
    if adj.shape != (n, n):
        raise ValueError(f"adj shape {adj.shape} does not match face count {n}")

    compatible = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            compatible[i, j] = adj[i, j] == 1 and is_complementary_pair(faces[i], faces[j], require_interval=False)
            compatible[j, i] = compatible[i, j]

    used = [False] * n
    groups: List[List[int]] = []

    quad_candidates = [i for i, face in enumerate(faces) if face["type"] in {"torus", "sphere"}]
    for group in compatible_components(compatible, quad_candidates):
        if len(group) != 4:
            continue
        if len({faces[i]["type"] for i in group}) != 1:
            continue
        edge_count = sum(1 for i in group for j in group if i < j and compatible[i, j])
        if edge_count < 3:
            continue
        for i in group:
            used[i] = True
        groups.append(group)

    for i in range(n):
        if used[i]:
            continue
        for j in range(i + 1, n):
            if used[j] or not compatible[i, j]:
                continue
            if is_complementary_pair(faces[i], faces[j]):
                used[i] = True
                used[j] = True
                groups.append([i, j])
                break

    groups.sort(key=lambda g: g[0])
    return groups


def align_face_groups(faces: List[Dict], groups: List[List[int]]) -> List[Dict]:
    aligned = copy.deepcopy(faces)
    for group in groups:
        group_faces = [aligned[i] for i in group]
        aligned_group = align_complementary_faces(group_faces)
        for idx, face in zip(group, aligned_group):
            aligned[idx] = face
    return aligned


def align_solid_surfs(
    solid_surfs: np.ndarray,
    solid_mask: np.ndarray,
    solid_adj: np.ndarray,
) -> Tuple[np.ndarray, List[List[int]], List[List[int]]]:
    aligned_surfs = np.array(solid_surfs, copy=True)
    edge_flags = [[0, 0, 0, 0] for _ in range(len(solid_surfs))]
    valid_indices, fitted_faces = fit_solid_faces(solid_surfs, solid_mask)
    if not fitted_faces:
        return aligned_surfs, [], edge_flags

    adj_valid = np.asarray(solid_adj, dtype=np.int32)[np.ix_(valid_indices, valid_indices)]
    groups = find_alignment_groups(fitted_faces, adj_valid)
    if not groups:
        return aligned_surfs, [], edge_flags

    aligned_faces = align_face_groups(fitted_faces, groups)
    edge_flags_valid = detect_aligned_edges(aligned_faces, adj_valid)
    for group in groups:
        for local_idx in group:
            source_idx = valid_indices[local_idx]
            points = np.asarray(aligned_faces[local_idx]["fit_points"], dtype=aligned_surfs.dtype)
            if points.shape != aligned_surfs[source_idx].shape:
                raise ValueError(f"aligned face shape {points.shape} does not match input shape {aligned_surfs[source_idx].shape}")
            aligned_surfs[source_idx] = points

    for local_idx, source_idx in enumerate(valid_indices):
        edge_flags[source_idx] = edge_flags_valid[local_idx]

    return aligned_surfs, [[valid_indices[i] for i in group] for group in groups], edge_flags


def extract_uv_aligned_edges(face, edge_flags):
    edge_flags = [int(x) for x in edge_flags] if isinstance(edge_flags, (list, tuple)) and len(edge_flags) == 4 else [0, 0, 0, 0]
    if not any(edge_flags):
        return []

    edges = get_edges_from_face(face)
    if not edges:
        return []

    surf = BRepAdaptor_Surface(face)
    u1, u2 = float(surf.FirstUParameter()), float(surf.LastUParameter())
    v1, v2 = float(surf.FirstVParameter()), float(surf.LastVParameter())
    du = max(abs(u2 - u1), 1e-9)
    dv = max(abs(v2 - v1), 1e-9)

    shape_analysis = ShapeAnalysis_Surface(BRep_Tool.Surface(face))
    edge_scores = []
    for edge in edges:
        curve, cmin, cmax = BRep_Tool.Curve(edge)
        if curve is None:
            continue
        scores = []
        for t in np.linspace(float(cmin), float(cmax), 9, dtype=float):
            try:
                p = gp_Pnt()
                curve.D0(float(t), p)
                uv = shape_analysis.ValueOfUV(p, 1e-5)
                uu, vv = float(uv.X()), float(uv.Y())
                scores.append([
                    abs(uu - u1) / du,
                    abs(uu - u2) / du,
                    abs(vv - v1) / dv,
                    abs(vv - v2) / dv,
                ])
            except Exception:
                continue
        if scores:
            edge_scores.append((edge, np.mean(np.asarray(scores, dtype=float), axis=0)))

    selected = []
    used = set()
    for target, flag in enumerate(edge_flags):
        if not flag:
            continue
        best_idx = -1
        best_cost = np.inf
        for idx, (edge, score) in enumerate(edge_scores):
            if idx in used:
                continue
            cost = float(score[target])
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
        if best_idx >= 0:
            used.add(best_idx)
            selected.append(edge_scores[best_idx][0])

    return selected
