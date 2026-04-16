"""
BEV 栅格 -> 相机视角稠密语义（地平面单应性 + 稠密逆映射最近邻采样）。
仅依赖 numpy，避免通过 bev_utils.__init__ 引入 torch/matplotlib 等。
与 project_reverse_test02 中 view / bev_shape 约定一致。
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

# 与 project_reverse_test02.py 一致
bev = {"h": 256, "w": 256, "h_meters": 50, "w_meters": 50, "offset": 0.51}
bev_shape = (256, 448)


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    sh = h / h_meters
    sw = w / w_meters
    return np.float32(
        [[0.0, -sw, w * offset + w / 2.0], [-sh, 0.0, h / 2.0], [0.0, 0.0, 1.0]]
    )


view = get_view_matrix(**bev)


def compute_bev_to_cam_homography(
    intrinsics: np.ndarray,
    E: np.ndarray,
    view_mat: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    p_cam ~ H @ p_bev，其中 p_bev 为 OpenCV BEV 像素齐次坐标 [x, y, 1]（x=列，y=行）。

    bev_pixels_to_cam 里用的是 V_inv @ [row, col, 1]（argwhere 顺序），即 V_inv @ [y, x, 1]，
    与 OpenCV 的 [x, y, 1] = [col, row, 1] 差一次交换，故右乘 P：[x,y,1] -> [y,x,1]。
    """
    if view_mat is None:
        view_mat = view
    V_inv = np.linalg.inv(np.asarray(view_mat, dtype=np.float64))
    S = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    # OpenCV (x,y) 与 numpy 索引 (row,col)：(x,y)=(col,row) -> [y,x,1]=[row,col,1]
    P_swap_xy = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    R = np.asarray(E, dtype=np.float64)[:3, :3]
    t = np.asarray(E, dtype=np.float64)[:3, 3:4]
    K = np.asarray(intrinsics, dtype=np.float64)
    H_ground_to_img = K @ np.hstack((R[:, :2], t))
    return H_ground_to_img @ S @ V_inv @ P_swap_xy


def bev_raster_warp_to_camera(
    bev_raster: np.ndarray,
    intrinsics: np.ndarray,
    E: np.ndarray,
    out_size: Optional[Tuple[int, int]] = None,
    interpolation: Optional[int] = None,
) -> np.ndarray:
    """
    稠密逆映射：对每个相机像素 (u,v) 计算 inv(H)@[u,v,1] 得到 BEV 亚像素位置，最近邻采样。
    不使用 cv2.warpPerspective：OpenCV 在 NEAREST 时对越界源坐标会钳位到图像边，
    会在投影域外产生错误的前景，语义上与「仅当逆映射落在 BEV 范围内才取值」不一致。
    """
    del interpolation  # 保留参数以兼容旧调用；实现固定为最近邻
    if out_size is None:
        out_h, out_w = int(bev_shape[0]), int(bev_shape[1])
    else:
        out_h, out_w = int(out_size[0]), int(out_size[1])
    if bev_raster.ndim != 3:
        raise ValueError(f"bev_raster 应为 (H,W,C)，当前 shape={bev_raster.shape}")
    bev_h, bev_w, C = bev_raster.shape

    x = np.asarray(bev_raster)
    if x.dtype != np.uint8:
        if np.issubdtype(x.dtype, np.floating) and x.max() <= 1.0 + 1e-6:
            x = (x > 0.5).astype(np.uint8) * 255
        else:
            x = (x > 0).astype(np.uint8) * 255
    else:
        x = np.where(x > 0, 255, 0).astype(np.uint8)

    H_fwd = compute_bev_to_cam_homography(intrinsics, E)
    M = np.linalg.inv(np.asarray(H_fwd, dtype=np.float64))

    # 相机像素 (u,v)=(列,行)，须与 im[v,u] 按 C 序展平一致：先行 v、行内 u 递增 → indexing='ij'
    vd = np.arange(out_h, dtype=np.float64)
    ud = np.arange(out_w, dtype=np.float64)
    V_grid, U_grid = np.meshgrid(vd, ud, indexing="ij")
    pts = np.stack([U_grid.ravel(), V_grid.ravel(), np.ones(U_grid.size, dtype=np.float64)], axis=0)
    src_h = M @ pts
    w_ = src_h[2, :]
    sx = (src_h[0] / w_).reshape(out_h, out_w)
    sy = (src_h[1] / w_).reshape(out_h, out_w)
    xi = np.rint(sx).astype(np.int32)
    yi = np.rint(sy).astype(np.int32)
    w_plane = w_.reshape(out_h, out_w)
    # 与地平面一侧一致：以下方中央为参考（通常对应路面，齐次 w 与路面同号）
    w_ref = float(w_plane[out_h - 1, out_w // 2])
    if abs(w_ref) < 1e-9:
        w_ref = float(np.median(w_plane[max(0, out_h - 4) :, :]))
    sign_ok = (w_plane * w_ref) > 0
    valid = sign_ok & (xi >= 0) & (xi < bev_w) & (yi >= 0) & (yi < bev_h)

    out = np.zeros((out_h, out_w, C), dtype=np.uint8)
    flat_idx = np.flatnonzero(valid.ravel())
    yi_v = yi.ravel()[flat_idx]
    xi_v = xi.ravel()[flat_idx]
    for c in range(C):
        ch = x[:, :, c]
        plane = np.zeros((out_h, out_w), dtype=np.uint8)
        pr = plane.ravel()
        pr[flat_idx] = ch[yi_v, xi_v]
        out[:, :, c] = (plane > 0).astype(np.uint8)
    return out


def bev_raster_warp_to_camera_rgb(
    bev_rgb: np.ndarray,
    intrinsics: np.ndarray,
    E: np.ndarray,
    out_size: Optional[Tuple[int, int]] = None,
    invalid_color: Union[Tuple[int, ...], Sequence[int]] = (0, 0, 0),
) -> np.ndarray:
    """
    RGB 标签图稠密逆映射：最近邻采样，**保留各通道颜色**，不做二值化。
    bev_rgb: (H, W, 3) 或 (H, W, 4)，推荐 uint8；float 且 max<=1 时按 0–255 量化。
    逆映射落在 BEV 外的相机像素填 invalid_color（长度不足 C 时右侧补 0）。
    """
    if out_size is None:
        out_h, out_w = int(bev_shape[0]), int(bev_shape[1])
    else:
        out_h, out_w = int(out_size[0]), int(out_size[1])
    if bev_rgb.ndim != 3 or bev_rgb.shape[2] not in (3, 4):
        raise ValueError(f"bev_rgb 应为 (H,W,3) 或 (H,W,4)，当前 shape={bev_rgb.shape}")

    x = np.asarray(bev_rgb)
    if np.issubdtype(x.dtype, np.floating):
        if x.max() <= 1.0 + 1e-6:
            x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        else:
            x = np.clip(x, 0, 255).astype(np.uint8)
    else:
        x = x.astype(np.uint8, copy=False)

    bev_h, bev_w, C = x.shape
    ic = np.asarray(invalid_color, dtype=np.uint8).ravel()
    if ic.size < C:
        ic = np.pad(ic, (0, C - ic.size), constant_values=0)
    else:
        ic = ic[:C]
    fill = ic.reshape(1, 1, C)

    H_fwd = compute_bev_to_cam_homography(intrinsics, E)
    M = np.linalg.inv(np.asarray(H_fwd, dtype=np.float64))

    vd = np.arange(out_h, dtype=np.float64)
    ud = np.arange(out_w, dtype=np.float64)
    V_grid, U_grid = np.meshgrid(vd, ud, indexing="ij")
    pts = np.stack([U_grid.ravel(), V_grid.ravel(), np.ones(U_grid.size, dtype=np.float64)], axis=0)
    src_h = M @ pts
    w_ = src_h[2, :]
    sx = (src_h[0] / w_).reshape(out_h, out_w)
    sy = (src_h[1] / w_).reshape(out_h, out_w)
    xi = np.rint(sx).astype(np.int32)
    yi = np.rint(sy).astype(np.int32)
    w_plane = w_.reshape(out_h, out_w)
    w_ref = float(w_plane[out_h - 1, out_w // 2])
    if abs(w_ref) < 1e-9:
        w_ref = float(np.median(w_plane[max(0, out_h - 4) :, :]))
    sign_ok = (w_plane * w_ref) > 0
    valid = sign_ok & (xi >= 0) & (xi < bev_w) & (yi >= 0) & (yi < bev_h)

    out = np.broadcast_to(fill, (out_h, out_w, C)).copy()
    flat_idx = np.flatnonzero(valid.ravel())
    yi_v = yi.ravel()[flat_idx]
    xi_v = xi.ravel()[flat_idx]
    out.reshape(-1, C)[flat_idx, :] = x[yi_v, xi_v, :]
    return out


def bev_raster_point_project_to_camera(
    bev_raster: np.ndarray,
    intrinsics: np.ndarray,
    E: np.ndarray,
    out_size: Optional[Tuple[int, int]] = None,
    near_z: float = 1e-3,
) -> np.ndarray:
    """
    与 `project_reverse_test02.bev_pixels_to_cam` + 针孔投影完全一致：对每个 BEV 前景点投影到相机平面，
    用于核对单应性 warp 与内外参是否正确（不做膨胀）。
    """
    if out_size is None:
        out_h, out_w = int(bev_shape[0]), int(bev_shape[1])
    else:
        out_h, out_w = int(out_size[0]), int(out_size[1])

    x = np.asarray(bev_raster)
    if x.dtype != np.uint8:
        if np.issubdtype(x.dtype, np.floating) and x.max() <= 1.0 + 1e-6:
            mask = x > 0.5
        else:
            mask = x > 0
    else:
        mask = x > 0

    V_inv = np.linalg.inv(np.asarray(view, dtype=np.float64))
    K = np.asarray(intrinsics, dtype=np.float64)
    E64 = np.asarray(E, dtype=np.float64)

    out = np.zeros((out_h, out_w, x.shape[2]), dtype=np.uint8)
    for c in range(x.shape[2]):
        ys, xs = np.where(mask[:, :, c])
        if len(xs) == 0:
            continue
        # 与 bev_pixels_to_cam 相同：齐次为 [row, col, 1]
        pix = np.stack([ys.astype(np.float64), xs.astype(np.float64), np.ones(len(xs))], axis=0)
        lc = (V_inv @ pix).T
        x_loc = lc[:, 1] / lc[:, 2]
        y_loc = lc[:, 0] / lc[:, 2]
        ones = np.ones(len(xs))
        local_4d = np.stack([x_loc, y_loc, np.zeros_like(x_loc), ones], axis=1)
        cam = (E64 @ local_4d.T).T[:, :3]
        in_front = cam[:, 2] > near_z
        cam = cam[in_front]
        proj = (K @ cam.T).T
        u = np.round(proj[:, 0] / proj[:, 2]).astype(np.int32)
        v = np.round(proj[:, 1] / proj[:, 2]).astype(np.int32)
        ok = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
        u, v = u[ok], v[ok]
        out[v, u, c] = 1
    return out
