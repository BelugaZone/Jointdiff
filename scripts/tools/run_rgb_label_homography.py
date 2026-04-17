"""
RGB 语义标签图（如可视化 PNG）-> 相机视角：地平面单应性 + 稠密逆映射 + 最近邻，保留 RGB。

与 `run_bev_homography_warp.py` 使用相同内外参；不做二值编码 decode，直接读 RGB(A)。

说明：
  - 输出分辨率固定为 bev_homography_warp.bev_shape = (高 256, 宽 448)，与编码标签投影一致；
    这是「宽扁」相机画布，不是 BEV 的正方形，并非宽高写反。
  - 地平面单应性下，相机图「上方」常对应天空/地平外，逆映射落不到 BEV 栅格内，故常见上缘一条
    为无效填充色，属正常现象；若需与 joint_gen/label 的位次完全一致，请加 --flip_bev_y。

用法（在 Jointdiff 仓库根目录下）:
  python scripts/tools/run_rgb_label_homography.py --input ../combined_811(1).png
  python scripts/tools/run_rgb_label_homography.py --input ../joint_gen/label_rgb/foo.png --flip_bev_y
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
from PIL import Image

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_bev_homography_module():
    path = os.path.join(str(_ROOT), "multi_view_generation", "bev_utils", "bev_homography_warp.py")
    spec = importlib.util.spec_from_file_location("bev_homography_warp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_hw = _load_bev_homography_module()
bev_raster_warp_to_camera_rgb = _hw.bev_raster_warp_to_camera_rgb
bev_shape = _hw.bev_shape

# 与 visual_label / run_bev_homography_warp 一致
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


def _default_joint_gen() -> str:
    return os.path.normpath(os.path.join(_ROOT, "..", "joint_gen"))


def main() -> None:
    joint = _default_joint_gen()
    parser = argparse.ArgumentParser(description="RGB 标签 BEV -> 相机（单应逆映射，保留颜色）")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 RGB 标签 PNG/JPG（建议与 BEV 栅格同约定尺寸，如 256×256）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(joint, "cam_label_homography_rgb"),
        help="输出目录",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default=None,
        help="输出文件名（默认 homography_rgb_<输入 basename>）",
    )
    parser.add_argument(
        "--flip_bev_y",
        action="store_true",
        help="对 BEV 做竖直 flipud，与 joint_gen/label 经 decode+transpose 后的朝向一致；"
        "多数单独导出的 RGB 语义图默认不需翻转",
    )
    parser.add_argument(
        "--keep_alpha",
        action="store_true",
        help="保留 alpha 为第 4 通道一并投影（否则只读 RGB）",
    )
    parser.add_argument(
        "--invalid_r",
        type=int,
        default=200,
        help="无效投影像素 R（默认 200 浅灰，接近常见背景）",
    )
    parser.add_argument(
        "--invalid_g",
        type=int,
        default=200,
        help="无效投影像素 G",
    )
    parser.add_argument(
        "--invalid_b",
        type=int,
        default=200,
        help="无效投影像素 B",
    )
    parser.add_argument(
        "--invalid_a",
        type=int,
        default=0,
        help="无效投影像素 A（仅 keep_alpha 时有效）",
    )
    args = parser.parse_args()

    inp = os.path.abspath(args.input)
    if not os.path.isfile(inp):
        raise SystemExit(f"找不到输入文件: {inp}")

    if args.keep_alpha:
        pil = Image.open(inp).convert("RGBA")
        bev_hwc = np.array(pil)
        invalid_color = (args.invalid_r, args.invalid_g, args.invalid_b, args.invalid_a)
    else:
        pil = Image.open(inp).convert("RGB")
        bev_hwc = np.array(pil)
        invalid_color = (args.invalid_r, args.invalid_g, args.invalid_b)

    if args.flip_bev_y:
        bev_hwc = np.flipud(bev_hwc)

    out = bev_raster_warp_to_camera_rgb(
        bev_hwc,
        intrinsics01,
        E01,
        invalid_color=invalid_color,
    )
    # 统计有效像素（非无效填充）
    ic = np.array(invalid_color[:3], dtype=np.int32).reshape(1, 1, 3)
    diff = np.abs(out.astype(np.int32) - ic) > 1
    valid_px = diff.any(axis=2).mean()

    if out.shape[2] == 4:
        out_pil = Image.fromarray(out, mode="RGBA")
    else:
        out_pil = Image.fromarray(out, mode="RGB")

    os.makedirs(args.out_dir, exist_ok=True)
    name = args.out_name or f"homography_rgb_{os.path.basename(inp)}"
    if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        name += ".png"
    out_path = os.path.join(args.out_dir, name)
    out_pil.save(out_path)
    print(
        "已保存:",
        os.path.abspath(out_path),
        "| 输出形状 H,W,C:",
        out.shape,
        f"| 与 bev_shape={bev_shape} 一致 (numpy 为 高×宽)",
    )
    print(
        f"非无效色像素占比约 {valid_px*100:.1f}%（地平面外/天空处为无效色属正常）"
    )


if __name__ == "__main__":
    main()
