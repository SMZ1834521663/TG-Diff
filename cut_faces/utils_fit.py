import traceback
import numpy as np
from scipy.spatial import cKDTree
from sklearn.covariance import MinCovDet
############################################### axis-aligned
def _canonicalize_axis_dir(axis_dir):
    """Fix axis sign ambiguity deterministically."""
    a = np.asarray(axis_dir, dtype=float).reshape(3)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    a = a / n
    k = int(np.argmax(np.abs(a)))
    if a[k] < 0:
        a = -a
    return a


def _uv_from_axis_preserve_sign(axis_dir):
    """Build deterministic (u_dir, v_dir) while keeping the input axis sign."""
    axis = np.asarray(axis_dir, dtype=float).reshape(3)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / n
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(ref, axis)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = ref - np.dot(ref, axis) * axis
    un = np.linalg.norm(u)
    if un < 1e-12:
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        u = ref - np.dot(ref, axis) * axis
        un = np.linalg.norm(u)
    u = u / (un + 1e-12)
    v = np.cross(axis, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return axis, u, v


def _angle_interval_from_samples(angles):
    """Get covered angular interval from scattered samples."""
    a = np.asarray(angles, dtype=float).reshape(-1)
    if a.size == 0:
        return -np.pi, np.pi
    a = np.mod(a, 2 * np.pi)
    a = np.sort(a)
    if a.size == 1:
        return float(a[0]), float(a[0] + 1e-6)
    diffs = np.diff(np.r_[a, a[0] + 2 * np.pi])
    k = int(np.argmax(diffs))
    start = float(a[(k + 1) % a.size])
    end = float(a[k])
    if end < start:
        end += 2 * np.pi
    if end - start > 2 * np.pi - 1e-4:
        end = start + 2 * np.pi
    return start, end


def _normalize_angle_interval(start, end):
    """Normalize start into [-pi, pi), keep span."""
    span = float(end - start)
    s = float(start)
    while s >= np.pi:
        s -= 2 * np.pi
    while s < -np.pi:
        s += 2 * np.pi
    return s, s + span

############################################### perturbation 
DEFAULT_PERTURB_EPS = 3e-5
def _uv_bump(shape, eps=DEFAULT_PERTURB_EPS):
    """Build a uv-grid bump whose value is zero on patch boundaries."""
    h, w = shape[:2]
    ii = np.linspace(-1.0, 1.0, h)
    jj = np.linspace(-1.0, 1.0, w)
    II, JJ = np.meshgrid(ii, jj, indexing="ij")
    return float(eps) * (1.0 - II**2) * (1.0 - JJ**2)

def _perturb_along(
    points,
    directions,
    eps=DEFAULT_PERTURB_EPS,
):
    dirs = np.asarray(directions, dtype=float)
    dirs = dirs / (np.linalg.norm(dirs, axis=2, keepdims=True) + 1e-12)
    bump = _uv_bump(points.shape, eps=eps)
    return points + bump[:, :, None] * dirs

def _perturb_plane(
    points,
    normal,
    eps=DEFAULT_PERTURB_EPS,
):
    direction = np.broadcast_to(np.asarray(normal, dtype=float).reshape(1, 1, 3), points.shape)
    return _perturb_along(points, direction, eps=eps)


def _perturb_radial(
    points,
    centerline_points,
    eps=DEFAULT_PERTURB_EPS,
):
    return _perturb_along(
        points,
        points - centerline_points,
        eps=eps,
    )


############################################### fit 
def fit_plane(points):
    H, W, _ = points.shape
    pts = points.reshape(-1, 3)
    n_u, n_v = H, W
    err_list=[]
    try:
        # Step 1: SVD fit for plane normal.
        center = np.mean(pts, axis=0, keepdims=True)
        X = pts - center
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        _, _, normal = Vh
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        normal = _canonicalize_axis_dir(normal)

        # Step 2: keep in-plane orientation from grid directions when possible.
        du = points[:, 1:, :] - points[:, :-1, :]
        du = du.reshape(-1, 3).mean(axis=0)
        du = du - np.dot(du, normal) * normal
        du_norm = np.linalg.norm(du)
        if du_norm < 1e-12:
            normal, u_dir, v_dir = _uv_from_axis_preserve_sign(normal)
        else:
            u_dir = du / du_norm
            v_dir = np.cross(normal, u_dir)
            v_dir = v_dir / (np.linalg.norm(v_dir) + 1e-12)
            dv = points[1:, :, :] - points[:-1, :, :]
            dv = dv.reshape(-1, 3).mean(axis=0)
            dv = dv - np.dot(dv, normal) * normal
            if np.dot(dv, v_dir) < 0:
                v_dir = -v_dir

        # Step 3: project all points into plane coordinates.
        vec = pts - center
        dist = np.dot(vec, normal.reshape(3,1))
        proj3d = pts - dist * normal.reshape(1,3)

        u = np.dot(proj3d - center, u_dir)
        v = np.dot(proj3d - center, v_dir)
        uv = np.stack([u, v], axis=1)
        umin, umax = uv[:, 0].min(), uv[:, 0].max()
        vmin, vmax = uv[:, 1].min(), uv[:, 1].max()

        # Step 4: sample fitted plane patch.
        u_vals = np.linspace(umin, umax, n_u)
        v_vals = np.linspace(vmin, vmax, n_v)
        uu, vv = np.meshgrid(u_vals, v_vals)
        uu = uu.flatten()
        vv = vv.flatten()
        grid_points = center + uu[:, None] * u_dir + vv[:, None] * v_dir
        sampled_points = grid_points.reshape(H, W, 3)
        sampled_points = _perturb_plane(sampled_points, normal)

        sucess = True
        err = evaluate_fit_error(points,sampled_points)

        # Return fitted parameters and boundary curves.
        sample_points_params={
            "normal":normal,
            "u_dir":u_dir,
            "v_dir":v_dir,
            "center":center,
            "u_min":umin,
            "u_max":umax,
            "v_min":vmin,
            "v_max":vmax,
            "e":[
                sampled_points[0, :, :],
                sampled_points[-1, :, :],
                sampled_points[:, 0, :],
                sampled_points[:, -1, :]
            ]
        }

        err_list.append([err,sampled_points,sucess,sample_points_params])

    except Exception as e:
        pass
        # traceback.print_exc()

    if len(err_list)>0:
        return_data = sorted(err_list,key=lambda x:x[0],reverse=False)[0]
        err,sampled_points,sucess,sample_points_params = return_data
        return sampled_points,err,sucess,sample_points_params
    else:
        return None,999,False,None
    


def fit_cylinder(points,decay=0):
    H, W, _ = points.shape
    pts = points.reshape(-1, 3)
    n_u, n_v = H, W
    err_list = []

    # Step 1: candidate axis directions from robust PCA and grid directions.
    mcd = MinCovDet(
        store_precision=True,
        assume_centered=False,
        support_fraction=None,
        random_state=1,
    ).fit(pts)
    cov = mcd.covariance_
    _, eigvecs = np.linalg.eigh(cov)
    axis_dirs = eigvecs

    col_dirs = points[-1, :, :] - points[0, :, :]
    axis_dir_new1 = np.mean(col_dirs, axis=0)
    axis_dir_new1 /= (np.linalg.norm(axis_dir_new1) + 1e-12)
    row_dirs = np.transpose(points, (1, 0, 2))[-1, :, :] - np.transpose(points, (1, 0, 2))[0, :, :]
    axis_dir_new2 = np.mean(row_dirs, axis=0)
    axis_dir_new2 /= (np.linalg.norm(axis_dir_new2) + 1e-12)
    axis_dirs = np.vstack([axis_dirs, axis_dir_new1.reshape(1, 3), axis_dir_new2.reshape(1, 3)])

    for axis_dir in axis_dirs:
        try:
            axis_dir = np.asarray(axis_dir, dtype=float)
            axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)
            axis_dir = _canonicalize_axis_dir(axis_dir)

            # Step 2: project points to axis coordinate.
            proj = np.dot(pts, axis_dir)
            min_proj, max_proj = proj.min(), proj.max()
            height = max_proj - min_proj
            if height < 1e-12:
                continue
            v_min, v_max = min_proj, max_proj

            atols = [0.05 * height, 0.1 * height, 0.15 * height, 0.8 * height]
            for atol in atols:
                try:
                    base_mask = np.isclose(proj, min_proj, atol=atol)
                    base_points = pts[base_mask]
                    if base_points.shape[0] < 3:
                        continue
                    base_center = base_points.mean(axis=0)

                    # Step 3: build local orthonormal frame from axis.
                    axis_dir, u, v = _uv_from_axis_preserve_sign(axis_dir)

                    # Step 4: fit base circle in local 2D coordinates.
                    pts_2d = np.dot(base_points - base_center, np.c_[u, v])
                    x, y = pts_2d[:, 0], pts_2d[:, 1]
                    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
                    b = x**2 + y**2
                    cx, cy, c3 = np.linalg.lstsq(A, b, rcond=None)[0]
                    radius = np.sqrt(max(c3 + cx**2 + cy**2, 1e-12))
                    bottom_center = base_center + cx * u + cy * v
                    center = bottom_center + 0.5 * height * axis_dir

                    # Step 5: robust angular interval from all points.
                    all_pts2d = np.dot(pts - base_center, np.c_[u, v])
                    angles = np.arctan2(all_pts2d[:, 1] - cy, all_pts2d[:, 0] - cx)
                    angle_min, angle_max = _angle_interval_from_samples(angles)
                    angle_min, angle_max = _normalize_angle_interval(angle_min, angle_max)

                    sucess = np.pi / 8 < (angle_max - angle_min) < 2 * np.pi + 1e-6
                    decay_value = decay * (angle_max - angle_min)
                    angle_min_decay, angle_max_decay = angle_min + decay_value, angle_max - decay_value

                    # Step 6: sample cylinder with and without decay.
                    u_angles = np.linspace(angle_min, angle_max, n_u)
                    v_heights = np.linspace(v_min, v_max, n_v)
                    U, Vh = np.meshgrid(u_angles, v_heights)
                    sampled_points = (
                        center[None, None, :]
                        + radius * np.cos(U[:, :, None]) * u[None, None, :]
                        + radius * np.sin(U[:, :, None]) * v[None, None, :]
                        + (Vh[:, :, None] - (v_min + 0.5 * height)) * axis_dir[None, None, :]
                    )

                    u_angles = np.linspace(angle_min_decay, angle_max_decay, n_u)
                    U, Vh = np.meshgrid(u_angles, v_heights)
                    sampled_points_decay = (
                        center[None, None, :]
                        + radius * np.cos(U[:, :, None]) * u[None, None, :]
                        + radius * np.sin(U[:, :, None]) * v[None, None, :]
                        + (Vh[:, :, None] - (v_min + 0.5 * height)) * axis_dir[None, None, :]
                    )
                    centerline = center[None, None, :] + (Vh[:, :, None] - (v_min + 0.5 * height)) * axis_dir[None, None, :]
                    sampled_points_decay = _perturb_radial(sampled_points_decay, centerline)

                    err = evaluate_fit_error(points, sampled_points)

                    sample_points_params = {
                        "center": center,
                        "radius": radius,
                        "axis_dir": axis_dir,
                        "u_dir": u,
                        "v_dir": v,
                        "u_min": angle_min,
                        "u_max": angle_max,
                        "v_min": v_min,
                        "v_max": v_max,
                        "height": height,
                        "e": [
                            sampled_points_decay[0, :, :],
                            sampled_points_decay[-1, :, :],
                            sampled_points_decay[:, 0, :],
                            sampled_points_decay[:, -1, :],
                        ],
                    }
                    err_list.append([err, sampled_points, sampled_points_decay, sucess, sample_points_params])
                except Exception:
                    pass
                    # traceback.print_exc()
        except Exception:
            pass
            # traceback.print_exc()

    if len(err_list) > 0:
        return_data = sorted(err_list, key=lambda x: x[0], reverse=False)[0]
        err, sampled_points, sampled_points_decay, sucess, sample_points_params = return_data
        return sampled_points_decay, err, sucess, sample_points_params
    else:
        return None, 999, False, None



def fit_cone(points,decay=0.0):
    H, W, _ = points.shape
    pts = points.reshape(-1,3)
    n_u, n_v = H, W
    err_list = []
    candidates_points = [points, np.transpose(points,(1,0,2))]
    for cand_points in candidates_points:
        try:
            # Step 1: fit local circles row-by-row and collect their 3D centers.
            local_centers = []
            local_radii = []

            for i in range(H):
                row_pts = cand_points[i,:,:]
                centroid = row_pts.mean(axis=0)
                _, _, Vt = np.linalg.svd(row_pts - centroid)
                u = Vt[0]
                v = Vt[1]

                pts2d = (row_pts - centroid) @ np.c_[u, v]
                x, y = pts2d[:,0], pts2d[:,1]

                A = np.c_[2*x, 2*y, np.ones_like(x)]
                b = x**2 + y**2
                try:
                    cx, cy, _ = np.linalg.lstsq(A,b,rcond=None)[0]
                    center_3d = centroid + cx*u + cy*v
                    local_centers.append(center_3d)
                    radii = np.linalg.norm(row_pts - center_3d, axis=1)
                    local_radii.append(radii.mean())
                except Exception:
                    continue

            local_centers = np.array(local_centers)
            local_radii = np.array(local_radii)
            if local_centers.shape[0] < 3:
                continue

            # Step 2: fit cone axis from local circle centers.
            base_center = local_centers.mean(axis=0)
            _, _, Vt = np.linalg.svd(local_centers - base_center)
            axis_dir = Vt[0]
            axis_dir /= (np.linalg.norm(axis_dir) + 1e-12)
            axis_dir = _canonicalize_axis_dir(axis_dir)

            # Step 3: project all points to axis coordinate and get height range.
            s_all = (pts - base_center) @ axis_dir
            s_min, s_max = s_all.min(), s_all.max()
            height = s_max - s_min
            if height < 1e-12:
                continue

            # Step 4: fit linear radius profile r = k*s + c.
            r = np.linalg.norm(pts - (base_center + np.outer(s_all, axis_dir)), axis=1)
            k, c = np.linalg.lstsq(np.vstack([s_all, np.ones_like(s_all)]).T, r, rcond=None)[0]
            # Keep axis sign fixed after k-sign handling, and build local frame.
            axis_dir, u, v = _uv_from_axis_preserve_sign(axis_dir)

            # Step 5: estimate angular interval and sample cone.
            atols = [0.1*height,0.05*height,0.15*height,0.8*height]
            for atol in atols:
                try:
                    base_mask = np.isclose(s_all, s_max, atol=atol)
                    base_pts = pts[base_mask]
                    if base_pts.shape[0] < 3:
                        continue

                    all_pts2d = (pts - base_center) @ np.c_[u, v]
                    angles = np.arctan2(all_pts2d[:, 1], all_pts2d[:, 0])
                    angle_min, angle_max = _angle_interval_from_samples(angles)
                    angle_min, angle_max = _normalize_angle_interval(angle_min, angle_max)

                    sucess = True
                    delta_r = abs(k * height) / (r.mean() + 1e-12)
                    if abs(k) < 0.05 or delta_r < 0.03:
                        sucess = False

                    decay_value = decay*(angle_max-angle_min)
                    angle_min_decay,angle_max_decay = angle_min+decay_value,angle_max-decay_value

                    U_grid, V_grid = np.meshgrid(
                        np.linspace(angle_min, angle_max, n_u),
                        np.linspace(s_min, s_max, n_v),
                    )
                    radii = np.maximum(k*V_grid + c, 1e-6)
                    sampled_points = (
                        base_center[None,None,:]
                        + V_grid[:,:,None]*axis_dir[None,None,:]
                        + radii[:,:,None] * (
                            np.cos(U_grid[:,:,None])*u[None,None,:]
                            + np.sin(U_grid[:,:,None])*v[None,None,:]
                        )
                    )

                    U_grid, V_grid = np.meshgrid(
                        np.linspace(angle_min_decay, angle_max_decay, n_u),
                        np.linspace(s_min, s_max, n_v),
                    )
                    radii = np.maximum(k*V_grid + c, 1e-6)
                    sampled_points_decay = (
                        base_center[None,None,:]
                        + V_grid[:,:,None]*axis_dir[None,None,:]
                        + radii[:,:,None] * (
                            np.cos(U_grid[:,:,None])*u[None,None,:]
                            + np.sin(U_grid[:,:,None])*v[None,None,:]
                        )
                    )
                    cone_axis_points = base_center[None, None, :] + V_grid[:, :, None] * axis_dir[None, None, :]
                    sampled_points_decay = _perturb_radial(sampled_points_decay, cone_axis_points)

                    err = evaluate_fit_error(pts,sampled_points)

                    sample_points_params={
                        "center":base_center,
                        "axis_dir":axis_dir,
                        "u_dir":u,
                        "v_dir":v,
                        "k":k,
                        "c":c,
                        "u_min":angle_min,
                        "u_max":angle_max,
                        "height":height,
                        "v_min":s_min,
                        "v_max":s_max,
                        "radius":abs(k * s_max + c),
                        "r":abs(k * s_min + c),
                        "e":[
                            sampled_points_decay[0, :, :],
                            sampled_points_decay[-1, :, :],
                            sampled_points_decay[:, 0, :],
                            sampled_points_decay[:, -1, :]
                        ]
                    }

                    err_list.append([err,sampled_points,sampled_points_decay,sucess,sample_points_params])
                except Exception:
                    pass
                    # traceback.print_exc()
        except Exception:
            pass
            # traceback.print_exc()

    if len(err_list)>0:
        return_data = sorted(err_list,key=lambda x:x[0],reverse=False)[0]
        err,sampled_points,sampled_points_decay,sucess,sample_points_params = return_data
        return sampled_points_decay,err,sucess,sample_points_params
    else:
        return None,999,False,None

def fit_torus(points,decay=[0.00,0.00]):
    H, W, _ = points.shape
    pts = points.reshape(-1, 3)
    n_u, n_v = W, H
    err_list = []
    candidates_points = [points, np.transpose(points, (1, 0, 2))]
    for cand_points in candidates_points:
        try:
            # Step 1: fit minor-circle centers column-by-column.
            small_centers = []
            small_radii = []
            for i in range(W):
                col_pts = cand_points[:, i, :]
                centroid = col_pts.mean(axis=0)
                _, _, Vt_ = np.linalg.svd(col_pts - centroid)
                u_plane, v_plane = Vt_[0], Vt_[1]
                pts2d = (col_pts - centroid) @ np.c_[u_plane, v_plane]
                x, y = pts2d[:, 0], pts2d[:, 1]
                A = np.c_[2 * x, 2 * y, np.ones_like(x)]
                b = x**2 + y**2
                try:
                    cx, cy, _ = np.linalg.lstsq(A, b, rcond=None)[0]
                    center_3d = centroid + cx * u_plane + cy * v_plane
                    small_centers.append(center_3d)
                    r_local = np.mean(np.linalg.norm(col_pts - center_3d, axis=1))
                    small_radii.append(r_local)
                except Exception:
                    continue
            small_centers = np.array(small_centers)
            if small_centers.shape[0] < 3:
                continue
            r = float(np.mean(small_radii))

            # Step 2: fit plane of minor-circle centers.
            plane_center = small_centers.mean(axis=0)
            _, _, Vt = np.linalg.svd(small_centers - plane_center)
            plane_normal = Vt[-1]

            # Step 3: estimate torus axis from row-wise local circle centers.
            local_centers = []
            for i in range(H):
                row_pts = cand_points[i, :, :]
                centroid = row_pts.mean(axis=0)
                _, _, Vt_ = np.linalg.svd(row_pts - centroid)
                u_plane, v_plane = Vt_[0], Vt_[1]
                pts2d = (row_pts - centroid) @ np.c_[u_plane, v_plane]
                x, y = pts2d[:, 0], pts2d[:, 1]
                A = np.c_[2 * x, 2 * y, np.ones_like(x)]
                b = x**2 + y**2
                try:
                    cx, cy, _ = np.linalg.lstsq(A, b, rcond=None)[0]
                    center_3d = centroid + cx * u_plane + cy * v_plane
                    local_centers.append(center_3d)
                except Exception:
                    continue
            local_centers = np.array(local_centers)
            if local_centers.shape[0] < 3:
                continue
            base_center = local_centers.mean(axis=0)
            _, _, Vt = np.linalg.svd(local_centers - base_center)
            axis_dir = Vt[0]
            axis_dir /= (np.linalg.norm(axis_dir) + 1e-12)
            axis_dir = _canonicalize_axis_dir(axis_dir)

            # Step 4: intersect axis line with minor-center plane to get torus center.
            denom = np.dot(axis_dir, plane_normal)
            if abs(denom) < 1e-12:
                continue
            t = np.dot(plane_center - base_center, plane_normal) / denom
            center = base_center + t * axis_dir

            # Step 5: build local frame from axis.
            axis_dir, u_vec, v_vec = _uv_from_axis_preserve_sign(axis_dir)

            # Step 6: estimate angular ranges u/v.
            X = (small_centers - center) @ u_vec
            Y = (small_centers - center) @ v_vec
            u_angle = np.arctan2(Y, X)
            u_min, u_max = _angle_interval_from_samples(u_angle)
            u_min, u_max = _normalize_angle_interval(u_min, u_max)

            v_angles = []
            for i in range(W):
                col_pts = cand_points[:, i, :]
                small_center = small_centers[i]
                pts_vecs = col_pts - small_center
                radial_vec = small_center - center
                radial_vec = radial_vec / (np.linalg.norm(radial_vec) + 1e-12)
                radial_proj = np.dot(pts_vecs, radial_vec)
                axis_proj = np.dot(pts_vecs, axis_dir)
                angles = np.arctan2(axis_proj, radial_proj)
                angles_unwrapped = np.unwrap(angles)

                if i > 0:
                    prev = v_angles[-1]
                    cur_min, cur_max = angles_unwrapped.min(), angles_unwrapped.max()
                    prev_min, prev_max = prev.min(), prev.max()
                    if cur_max < prev_min:
                        angles_unwrapped += 2 * np.pi
                    elif cur_min > prev_max:
                        angles_unwrapped -= 2 * np.pi

                v_angles.append(angles_unwrapped)

            v_angles_all = np.concatenate(v_angles) if len(v_angles) > 0 else np.array([0.0, 2.0 * np.pi])
            v_min, v_max = _angle_interval_from_samples(v_angles_all)
            v_min, v_max = _normalize_angle_interval(v_min, v_max)

            sucess = (u_max - u_min > np.pi / 8) and (v_max - v_min > np.pi / 8)
            decay_value = decay[0] * (u_max - u_min)
            u_min_decay, u_max_decay = u_min + decay_value, u_max - decay_value
            decay_value = decay[1] * (v_max - v_min)
            v_min_decay, v_max_decay = v_min + decay_value, v_max - decay_value

            # Step 7: major radius.
            R = np.mean(np.linalg.norm(small_centers - center, axis=1))

            # Step 8: sample torus with and without decay.
            full_circle = np.isclose(u_max - u_min, 2 * np.pi)
            u = np.linspace(u_min, u_max, n_u, endpoint=not full_circle)
            v = np.linspace(v_min, v_max, n_v)
            U, V = np.meshgrid(u, v)
            X = (R + r * np.cos(V)) * np.cos(U)
            Y = (R + r * np.cos(V)) * np.sin(U)
            Z = r * np.sin(V)
            sampled_local = np.stack([X, Y, Z], axis=2)
            pts_local = sampled_local.reshape(-1, 3)
            R_mat = np.stack([u_vec, v_vec, axis_dir], axis=1)
            sampled_points = (R_mat @ pts_local.T).T + center
            sampled_points = sampled_points.reshape(n_v, n_u, 3)

            full_circle = np.isclose(u_max_decay - u_min_decay, 2 * np.pi)
            u = np.linspace(u_min_decay, u_max_decay, n_u, endpoint=not full_circle)
            v = np.linspace(v_min_decay, v_max_decay, n_v)
            U, V = np.meshgrid(u, v)
            X = (R + r * np.cos(V)) * np.cos(U)
            Y = (R + r * np.cos(V)) * np.sin(U)
            Z = r * np.sin(V)
            sampled_local = np.stack([X, Y, Z], axis=2)
            pts_local = sampled_local.reshape(-1, 3)
            sampled_points_decay = (R_mat @ pts_local.T).T + center
            sampled_points_decay = sampled_points_decay.reshape(n_v, n_u, 3)
            normal_local = np.stack([np.cos(V) * np.cos(U), np.cos(V) * np.sin(U), np.sin(V)], axis=2)
            normal_world = (R_mat @ normal_local.reshape(-1, 3).T).T.reshape(n_v, n_u, 3)
            sampled_points_decay = _perturb_along(sampled_points_decay, normal_world)

            err = evaluate_fit_error(pts, sampled_points)

            sample_points_params = {
                "center": center,
                "axis_dir": axis_dir,
                "u_dir": u_vec,
                "v_dir": v_vec,
                "u_min": u_min,
                "u_max": u_max,
                "v_min": v_min,
                "v_max": v_max,
                "radius": R,
                "r": r,
                "e": [
                    sampled_points_decay[0, :, :],
                    sampled_points_decay[-1, :, :],
                    sampled_points_decay[:, 0, :],
                    sampled_points_decay[:, -1, :],
                ],
            }
            err_list.append([err, sampled_points, sampled_points_decay, sucess, sample_points_params])
        except Exception:
            pass
            # traceback.print_exc()

    if len(err_list) > 0:
        return_data = sorted(err_list, key=lambda x: x[0], reverse=False)[0]
        err, sampled_points, sampled_points_decay, sucess, sample_points_params = return_data
        return sampled_points_decay, err, sucess, sample_points_params
    else:
        return None, 999, False, None


def fit_sphere(points, decay=[0.0,0.0]):
    H, W, _ = points.shape
    pts = points.reshape(-1, 3)
    n_u, n_v = W, H
    err_list = []
    candidates_points = [points, np.transpose(points, (1, 0, 2))]
    for cand_points in candidates_points:
        try:
            # Step 1: fit sphere center and radius.
            X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
            A = np.c_[2 * X, 2 * Y, 2 * Z, np.ones_like(X)]
            b = X**2 + Y**2 + Z**2
            cx, cy, cz, D = np.linalg.lstsq(A, b, rcond=None)[0]
            center = np.array([cx, cy, cz])
            radius = np.sqrt(max(D + cx**2 + cy**2 + cz**2, 1e-12))

            # Step 2: estimate stable polar axis from row-wise local circle centers.
            local_centers = []
            for i in range(H):
                row_pts = cand_points[i, :, :]
                centroid = row_pts.mean(axis=0)
                _, _, Vt_ = np.linalg.svd(row_pts - centroid)
                u_plane, v_plane = Vt_[0], Vt_[1]
                pts2d = (row_pts - centroid) @ np.c_[u_plane, v_plane]
                x, y = pts2d[:, 0], pts2d[:, 1]
                A = np.c_[2 * x, 2 * y, np.ones_like(x)]
                b = x**2 + y**2
                try:
                    cx, cy, _ = np.linalg.lstsq(A, b, rcond=None)[0]
                    center_3d = centroid + cx * u_plane + cy * v_plane
                    local_centers.append(center_3d)
                except Exception:
                    continue
            local_centers = np.array(local_centers)
            if local_centers.shape[0] < 3:
                continue
            base_center = local_centers.mean(axis=0)
            _, _, Vt = np.linalg.svd(local_centers - base_center)
            z_axis = Vt[0]
            z_axis /= (np.linalg.norm(z_axis) + 1e-12)
            z_axis = _canonicalize_axis_dir(z_axis)

            # Step 3: build local frame from polar axis.
            z_axis, x_axis, y_axis = _uv_from_axis_preserve_sign(z_axis)

            # Step 4: estimate angular ranges.
            vecs = cand_points - center
            r_len = np.linalg.norm(vecs, axis=2)
            x_l = vecs @ x_axis
            y_l = vecs @ y_axis
            z_l = vecs @ z_axis

            u_all = np.arctan2(y_l.reshape(-1), x_l.reshape(-1))
            u_min, u_max = _angle_interval_from_samples(u_all)
            u_min, u_max = _normalize_angle_interval(u_min, u_max)

            cos_v = z_l / (r_len + 1e-12)
            cos_v = np.clip(cos_v, -1.0, 1.0)
            v_angle = np.arccos(cos_v)
            v_min, v_max = v_angle.min(), v_angle.max()

            sucess = False
            if u_max - u_min > np.pi / 6 and v_max - v_min > np.pi / 6 and 1 / 3 < ((u_max - u_min) / ((v_max - v_min) * 2)) < 3:
                sucess = True

            decay_value = decay[0] * (u_max - u_min)
            u_min_decay, u_max_decay = u_min + decay_value, u_max - decay_value
            decay_value = decay[1] * (v_max - v_min)
            v_min_decay, v_max_decay = v_min + decay_value, v_max - decay_value

            # Step 5: sample sphere patch with and without decay.
            R = np.column_stack([x_axis, y_axis, z_axis])

            uu = np.linspace(u_min, u_max, n_u)
            vv = np.linspace(v_min, v_max, n_v)
            U, V = np.meshgrid(uu, vv)
            Xs = radius * np.sin(V) * np.cos(U)
            Ys = radius * np.sin(V) * np.sin(U)
            Zs = radius * np.cos(V)
            grid_local = np.stack([Xs, Ys, Zs], axis=2)
            sampled_points = grid_local.reshape(-1, 3) @ R.T
            sampled_points = sampled_points.reshape(n_v, n_u, 3) + center

            uu = np.linspace(u_min_decay, u_max_decay, n_u)
            vv = np.linspace(v_min_decay, v_max_decay, n_v)
            U, V = np.meshgrid(uu, vv)
            Xs = radius * np.sin(V) * np.cos(U)
            Ys = radius * np.sin(V) * np.sin(U)
            Zs = radius * np.cos(V)
            grid_local = np.stack([Xs, Ys, Zs], axis=2)
            sampled_points_decay = grid_local.reshape(-1, 3) @ R.T
            sampled_points_decay = sampled_points_decay.reshape(n_v, n_u, 3) + center
            normal_world = (grid_local / (radius + 1e-12)).reshape(-1, 3) @ R.T
            normal_world = normal_world.reshape(n_v, n_u, 3)
            sampled_points_decay = _perturb_along(sampled_points_decay, normal_world)

            err = evaluate_fit_error(pts, sampled_points)

            sample_points_params = {
                "center": center,
                "axis_dir": z_axis,
                "u_dir": x_axis,
                "v_dir": y_axis,
                "u_min": u_min,
                "u_max": u_max,
                "v_min": v_min,
                "v_max": v_max,
                "radius": radius,
                "e": [
                    sampled_points_decay[0, :, :],
                    sampled_points_decay[-1, :, :],
                    sampled_points_decay[:, 0, :],
                    sampled_points_decay[:, -1, :],
                ],
            }

            err_list.append([err, sampled_points, sampled_points_decay, sucess, sample_points_params])
        except Exception:
            pass
            # traceback.print_exc()

    if len(err_list) > 0:
        return_data = sorted(err_list, key=lambda x: x[0], reverse=False)[0]
        err, sampled_points, sampled_points_decay, sucess, sample_points_params = return_data
        return sampled_points_decay, err, sucess, sample_points_params
    else:
        return None, 999, False, None


def evaluate_fit_error(orig_points, fit_points, boundary_weight=3.0):

    if len(fit_points.shape)==3:
        H, W, _ = fit_points.shape
    else:
        H = W = fit_points.shape[0]**0.5

    orig_points = orig_points.reshape(-1,3)
    fit_points = fit_points.reshape(-1,3)

    min_coords = orig_points.min(axis=0)
    max_coords = orig_points.max(axis=0)
    bbox_size = max_coords - min_coords
    diag_len = np.linalg.norm(bbox_size)

    weight = np.ones((H, W))
    weight[0, :]   = boundary_weight      # top
    weight[-1, :]  = boundary_weight      # bottom
    weight[:, 0]   = boundary_weight      # left
    weight[:, -1]  = boundary_weight      # right
    weight = weight.reshape(-1)

    tree_fit = cKDTree(fit_points)
    d1, _ = tree_fit.query(orig_points)
    tree_orig = cKDTree(orig_points)
    d2, _ = tree_orig.query(fit_points)

    mean_err = 0.5 * (np.sum(d1 * weight) + np.sum(d2 * weight)) / (np.sum(weight) * diag_len)
    return mean_err


