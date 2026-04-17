"""
BEV 栅格标签 -> 相机视角稠密语义（单应性 inverse warp / warpPerspective）。

不经过 multi_view_generation.bev_utils 包初始化，避免拉取 torch/matplotlib 等（与损坏的 NumPy 环境冲突）。
仅加载 bev_homography_warp.py（numpy + cv2）。

用法（在 Jointdiff 仓库根目录下）:
  python scripts/tools/run_bev_homography_warp.py
  python scripts/tools/run_bev_homography_warp.py --method point --out_dir_point ../joint_gen/cam_label_point_proj
  python scripts/tools/run_bev_homography_warp.py --method both --limit 10
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_bev_homography_module():
    path = os.path.join(
        str(_ROOT), "multi_view_generation", "bev_utils", "bev_homography_warp.py"
    )
    spec = importlib.util.spec_from_file_location("bev_homography_warp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_hw = _load_bev_homography_module()
bev_raster_warp_to_camera = _hw.bev_raster_warp_to_camera
bev_raster_point_project_to_camera = _hw.bev_raster_point_project_to_camera

# 与 visual_label.py 一致
E01 = np.array(
    [
        [5.6847786e-03, -9.9998349e-01, 8.0507132e-04, 5.0603189e-03],
        [-5.6366678e-03, -8.3711528e-04, -9.9998379e-01, 1.5205332e00],
        [9.9996793e-01, 5.6801485e-03, -5.6413338e-03, -1.6923035e00],
        [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
    ]
)

intrinsics01 = np.array(
    [
        [350.7877, 0.0, 231.4447],
        [0.0, 356.3557, 133.6845],
        [0.0, 0.0, 1.0],
    ]
)


def encode_binary_labels(masks: np.ndarray) -> np.ndarray:
    bits = np.power(2, np.arange(len(masks), dtype=np.int32))
    return (masks.astype(np.int32) * bits.reshape(-1, 1, 1)).sum(0)


def decode_binary_labels_np(encoded_2d: np.ndarray, nclass: int) -> np.ndarray:
    """返回 (nclass, H, W) bool，与 torch 版 decode 等价。"""
    bits = (2 ** np.arange(nclass, dtype=np.int64)).reshape(-1, 1, 1)
    return ((encoded_2d.astype(np.int64)[None, ...] & bits) > 0)


# 与 visualize.COLORS 中前四类静态地物一致（用于 4 通道可视化）
_VIZ_RGB = np.array(
    [
        [90, 90, 90],
        [255, 255, 255],
        [46, 139, 87],
        [148, 0, 211],
    ],
    dtype=np.float32,
)
_NOTHING = np.array([200, 200, 200], dtype=np.float32)


def viz_bev_hwc_float(bev_hwc: np.ndarray) -> Image.Image:
    """bev (H,W,C) float 0/1 -> RGB PIL，逻辑接近 viz_bev（多通道 argmax）。"""
    bev = np.asarray(bev_hwc, dtype=np.float32)
    if bev.max() > 1.0:
        bev = bev / 255.0
    h, w, c = bev.shape
    n = min(c, len(_VIZ_RGB))
    colors = _VIZ_RGB[:n]
    eps = (1e-5 * np.arange(n))[None, None, :]
    sl = bev[:, :, :n] + eps
    idx = np.argmax(sl, axis=-1)
    val = np.take_along_axis(bev[:, :, :n], idx[..., None], axis=-1)[..., 0]
    rgb = val[..., None] * colors[idx] + (1.0 - val[..., None]) * _NOTHING
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def _default_joint_gen_root() -> str:
    return os.path.normpath(os.path.join(_ROOT, "..", "joint_gen"))


def _write_synthetic_bev_png(path: str, size: int = 256) -> None:
    masks = np.zeros((6, size, size), dtype=bool)
    masks[0, size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = True
    masks[1, 10:30, 10:80] = True
    enc = encode_binary_labels(masks)
    Image.fromarray(enc.astype(np.int32)).save(path)


def main() -> None:
    joint = _default_joint_gen_root()
    parser = argparse.ArgumentParser(description="BEV 栅格 -> 相机稠密投影（单应性）")
    parser.add_argument(
        "--bev_dir",
        type=str,
        default=os.path.join(joint, "label"),
        help="BEV 标签目录（PNG，与 visual_label 相同编码）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(joint, "cam_label_homography"),
        help="单应性 warp 输出目录（method 含 homography 时）",
    )
    parser.add_argument(
        "--out_dir_point",
        type=str,
        default=os.path.join(joint, "cam_label_point_proj"),
        help="点投影输出目录（method 含 point 时）",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=("homography", "point", "both"),
        default="homography",
        help="homography=单应稠密 warp；point=与 bev_pixels_to_cam 一致的逐点投影；both=两者都写并打印差异",
    )
    parser.add_argument("--nclass", type=int, default=6, help="编码通道数")
    parser.add_argument(
        "--channels",
        type=int,
        default=4,
        help="参与 warp 的前 C 个语义通道",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="只处理前 N 张（按文件名排序后截取），默认全部",
    )
    args = parser.parse_args()

    os.makedirs(args.bev_dir, exist_ok=True)
    if args.method in ("homography", "both"):
        os.makedirs(args.out_dir, exist_ok=True)
    if args.method in ("point", "both"):
        os.makedirs(args.out_dir_point, exist_ok=True)

    bev_paths = sorted(glob.glob(os.path.join(args.bev_dir, "*.png")))
    if len(bev_paths) == 0:
        demo = os.path.join(args.bev_dir, "_synthetic_demo.png")
        _write_synthetic_bev_png(demo)
        bev_paths = [demo]
        print(f"未找到标签 PNG，已写入示例: {demo}")
    elif args.limit is not None and args.limit > 0:
        bev_paths = bev_paths[: args.limit]

    desc = {"homography": "homography warp", "point": "point project", "both": "warp+point"}[
        args.method
    ]
    for idx, bev_path in enumerate(tqdm(bev_paths, desc=desc)):
        img_array = np.array(Image.open(bev_path)).astype(np.int32)
        binary_labels = decode_binary_labels_np(img_array, args.nclass)

        bev_ch = binary_labels[: args.channels, :, :].astype(np.float32)
        bev_hwc = np.transpose(bev_ch, (1, 2, 0))
        bev_hwc = np.flipud(bev_hwc)

        filename = os.path.basename(bev_path)

        bev_cam_h = bev_cam_p = None
        if args.method in ("homography", "both"):
            bev_cam_h = bev_raster_warp_to_camera(bev_hwc, intrinsics01, E01)
            viz_bev_hwc_float(bev_cam_h.astype(np.float32)).save(
                os.path.join(args.out_dir, f"homography_{filename}")
            )

        if args.method in ("point", "both"):
            bev_cam_p = bev_raster_point_project_to_camera(bev_hwc, intrinsics01, E01)
            viz_bev_hwc_float(bev_cam_p.astype(np.float32)).save(
                os.path.join(args.out_dir_point, f"point_{filename}")
            )

        if args.method == "both" and idx == 0 and bev_cam_h is not None and bev_cam_p is not None:
            h, p = bev_cam_h, bev_cam_p
            inter = (h > 0) & (p > 0)
            union = (h > 0) | (p > 0)
            iou = inter.sum() / (union.sum() + 1e-8)
            print(
                "首张图 homography vs point IoU(二值):",
                float(iou),
                "| 仅 homography:",
                int((h > 0).sum()),
                "| 仅 point:",
                int((p > 0).sum()),
            )

    out_msg = []
    if args.method in ("homography", "both"):
        out_msg.append("单应: " + os.path.abspath(args.out_dir))
    if args.method in ("point", "both"):
        out_msg.append("点投影: " + os.path.abspath(args.out_dir_point))
    print("完成。共处理", len(bev_paths), "张。", " ".join(out_msg))


if __name__ == "__main__":
    main()
