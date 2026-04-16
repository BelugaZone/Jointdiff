import pyrootutils
from typing import Union
from image_utils import Im
from multi_view_generation.bev_utils.util import Cameras
import multi_view_generation.bev_utils.util as util
from multi_view_generation.bev_utils.nuscenes_helper import CLASSES , encode , decode
from multi_view_generation.bev_utils import Dataset
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from einops import repeat
import os

# Taken from:
# https://github.com/bradyz/cross_view_transformers/blob/master/cross_view_transformer/visualizations/common.py
# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/color_map.py
# COLORS = {
#     # static
#     "lane": (110, 110, 110),
#     "road_segment": (90, 90, 90),
#     # dividers
#     "road_divider": (255, 200, 0),
#     "lane_divider": (130, 130, 130),
#     # dynamic
#     "car": (255, 158, 0),
#     "truck": (255, 99, 71),
#     "bus": (255, 127, 80),
#     "trailer": (255, 140, 0),
#     "construction": (233, 150, 70),
#     "pedestrian": (0, 0, 230),
#     "motorcycle": (255, 61, 99),
#     "bicycle": (220, 20, 60),
#     "nothing": (200, 200, 200),
# }
COLORS = {
    # static - 使用冷色调为主，保持灰度/蓝色调
    "drivable_area": (90, 90, 90),  # 淡蓝色（可行驶区域）
    # "road_segment": (90, 90, 90),      # 深灰色（已有）
    "lane": (110, 110, 110),            # 灰色（已有）
    "ped_crossing": (255, 255, 255),    # 白色（斑马线）
    "walkway": (46, 139, 87),           # 森林绿（人行道）
    "carpark_area": (148, 0, 211),     # 紫罗兰色（停车场）
    # "stop_line": (255, 0, 0),           # 红色（停止线）
    
    # divider - 使用高对比度颜色
    "road_divider": (255, 200, 0),      # 金色（已有）
    
    # dynamic - 保持现有暖色调
    "car": (255, 158, 0),              # 橙色（已有）
    "truck": (255, 99, 71),            # 番茄红（已有）
    "bus": (255, 127, 80),             # 珊瑚红（已有）
    "trailer": (255, 140, 0),          # 深橙色（已有）
    "construction": (233, 150, 70),    # 棕褐色（已有）
    "pedestrian": (0, 0, 230),         # 深蓝色（已有）
    "motorcycle": (255, 61, 99),       # 粉红色（已有）
    "bicycle": (220, 20, 60),          # 猩红色（已有）
    
    "nothing": (200, 200, 200)         # 浅灰色（已有）
}

Three_C_COLORS= {
    "road_segment": (90, 90, 90),
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "nothing": (200, 200, 200),
    
}

NUSCENES_COLORS = {
    "drivable_area": (0, 255, 0),  # green
    "ped_crossing": (255, 105, 180),  # hotpink
    "walkway": (255, 215, 0),  # gold
    "carpark_area": (255, 165, 0),  # orange
    "bus": (255, 127, 80),  # coral
    "bicycle": (220, 20, 60),  # crimson
    "car": (255, 158, 0),  # orange
    "construction_vehicle": (233, 150, 70),  # darkorange
    "motorcycle": (255, 61, 99),  # deep pink
    "trailer": (255, 140, 0),  # darkorange
    "truck": (255, 99, 71),  # tomato
    "pedestrian": (0, 0, 230),  # blue
    "traffic_cone": (255, 165, 0),  # orange
    "barrier": (105, 105, 105),  # dimgray
    "nothing": (200, 200, 200),
}

ARGOVERSE_COLORS = {
    "driveable_area": (110, 110, 110),
    "lane_divider": (130, 130, 130),
    "ped_xing": (255, 200, 0),
    "pedestrian": (0, 0, 230),
    "vehicle": (255, 158, 0),
    "large_vehicle": (255, 99, 71),
    "other": (255, 127, 80),
    "nothing": (200, 200, 200),
}

def save_binary_as_image(data, filename):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    os.makedirs(Path(filename).parents[0], exist_ok=True)
    plt.imsave(filename, data, cmap=cm.hot, vmin=0, vmax=1)

def return_binary_as_image(data):
    return repeat(((data.detach().cpu().numpy()) * 255).astype(np.uint8), '... -> ... c', c=3)

def viz_bev(bev: Union[np.ndarray, torch.FloatTensor], dataset: Dataset = Dataset.NUSCENES) -> Im:
    """(c h w) torch [0, 1] float -> (h w 3) np [0, 255] uint8"""
    
    if torch.is_tensor(bev):
        bev = bev.detach().cpu().numpy()

    
    if bev.max() > 1:
        bev = (bev / 255.0)
    
    bev = bev.astype(np.float32)
    assert len(bev.shape) == 3
    assert bev.dtype == np.float32
    if bev.shape[1] == bev.shape[2] and bev.shape[0] < bev.shape[1]:
        bev = bev.transpose(1, 2, 0)

    # img is (h w c) np [0, 1] float
    if dataset == Dataset.NUSCENES:
        color_dict = COLORS
        classes = CLASSES
        bev = bev[..., :12]
    elif dataset == Dataset.ARGOVERSE:
        color_dict = ARGOVERSE_COLORS
        classes = ["driveable_area", "lane_divider", "ped_xing", "other", "pedestrian", "vehicle", "large_vehicle"]
        bev[..., range(bev.shape[-1])] = bev[..., [4, 5, 6, 3, 1, 0, 2]]
    else:
        raise ValueError()

    # print("bev.min() =", bev.min())
    # print("bev.max() =", bev.max())
    assert 0 <= bev.min() <= bev.max() <= 1.0
    colors = np.array([color_dict[s] for s in classes], dtype=np.uint8)

    h, w, c = bev.shape

    # assert c == len(classes)

    # Prioritize higher class labels
    eps = (1e-5 * np.arange(c))[None, None]
    idx = (bev + eps).argmax(axis=-1)
    val = np.take_along_axis(bev, idx[..., None], -1)

    # Spots with no labels are light grey
    empty = np.uint8(color_dict["nothing"])[None, None]

    result = (val * colors[idx]) + ((1 - val) * empty)
    result = np.uint8(result)

    return Im(result)

def vis_singlec( bev, save_path):
        
        bev = bev.astype(np.uint8)
        
        new_array = encode(bev)
        
        vis_array = new_array.astype(np.uint8)
        # unique_values = np.unique(vis_array)

        # # 打印结果
        # print(f"Unique label values ({len(unique_values)} classes):")
        # print(np.sort(unique_values))

        label_img = Image.fromarray(vis_array, mode='L')

        # 创建调色板（256种颜色，每个颜色3字节RGB值）
        palette = [0] * 768  # 初始化全黑调色板

        # 定义颜色映射（索引 -> RGB）
        palette[0 * 3:1 * 3] = [0, 0, 0]        # 0: 黑色（背景）
        palette[1 * 3:2 * 3] = [255, 0, 0]      # 1: 红色
        palette[2 * 3:3 * 3] = [0, 255, 0]      # 2: 绿色
        palette[3 * 3:4 * 3] = [0, 0, 255]      # 3: 蓝色
        palette[4 * 3:5 * 3] = [255, 255, 0]    # 4: 黄色
        palette[5 * 3:6 * 3] = [128, 0, 128]    # 5: 品红
        palette[6 * 3:7 * 3] = [0, 255, 255]    # 6: 青色
        palette[7 * 3:8 * 3] = [255, 0, 128]    # 7: 紫色
        palette[8 * 3:9 * 3] = [255, 165, 0]    # 8: 橙色


        # 转换为调色板模式并应用颜色
        label_p = label_img.convert('P')
        label_p.putpalette(palette)

        # 保存可视化结果
        label_p.convert('RGB').save(save_path)
def hex_to_rgb(hex_color):
    """将十六进制颜色代码转换为RGB元组"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def visualize_map_mask(map_mask, output_path='output.png'):
    color_map = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99',
                 '#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a',
                 '#7e772e','#00ff00','#0000ff','#00ffff','#303030']
    ori_shape = map_mask.shape
    vis = np.zeros((ori_shape[1], ori_shape[2], 3), dtype=np.uint8)
    vis_flat = vis.reshape(-1, 3)
    map_mask_flat = map_mask.reshape(ori_shape[0], -1)
    
    for layer_id in range(map_mask_flat.shape[0]):
        keep = np.where(map_mask_flat[layer_id, :])[0]
        rgb = hex_to_rgb(color_map[layer_id])
        # 将RGB分量直接分配到对应通道
        vis_flat[keep, 0] = rgb[0]  # Red通道
        vis_flat[keep, 1] = rgb[1]  # Green通道
        vis_flat[keep, 2] = rgb[2]  # Blue通道
    
    vis = vis_flat.reshape(ori_shape[1], ori_shape[2], 3)
    Image.fromarray(vis).save(output_path)  # 保存为PNG

def raw_output_data_bev_grid(batch):
    images = Im(batch['image']).denormalize().torch
    ret_images = []

    for i in range(batch['image'].shape[0]):
        if batch['dataset'][i] == 'nuscenes':
            viz_func = camera_bev_grid
        else:
            viz_func = argoverse_camera_bev_grid
        image_dict = {batch['cam_name'][k][i]: images[i, k] for k in range(images.shape[1])}
        ret_images.append(viz_func(images=image_dict, bev=batch["segmentation"][i]))
    return ret_images


def batched_camera_bev_grid(cfg, images, bev=None, pred=None):
    if len(images.shape) == 4:
        images = images.unsqueeze(0)

    image_names = cfg.cam_names
    if cfg.dataset == Dataset.NUSCENES:
        viz_func = camera_bev_grid
        image_names = Cameras.NUSCENES_CAMERAS if images.shape[1] == 6 else Cameras.NUSCENES_ABLATION_CAMERAS
    else:
        viz_func = argoverse_camera_bev_grid
        # image_names = Cameras.ARGOVERSE_CAMERAS
        
    ret_images = []
    for i in range(images.shape[0]):
        image_dict = {image_names[k]: images[i, k] for k in range(images.shape[1])}
        ret_images.append(Im(viz_func(image_dict, bev[i] if bev is not None else None, pred[i] if pred is not None else None)).torch)

    return torch.stack(ret_images, dim=0)


def ground_truth_camera_bev_grid(images: dict, bev=None, pred=None, keep_bev_space=False, add_car=True):
    images = {k: (Im(v).pil if not isinstance(v, Image.Image) else v) for k, v in images.items()}

    landscape_width = images[next(iter(images))].size[0]
    landscape_height = images[next(iter(images))].size[1]

    horiz_padding = 0
    vert_padding = 5

    six_cameras = len(images) == 6
    height = 2 * landscape_height + 2 * vert_padding + landscape_width
    width = landscape_width + landscape_width + landscape_width + 3 * horiz_padding + (landscape_height * 2 + horiz_padding if pred is not None else 0)
    pred_width = 0

    # bev = None

    if bev is not None:
        bev = viz_bev(bev, dataset=Dataset.NUSCENES).pil.resize((landscape_width, landscape_width))
    elif keep_bev_space:
        bev = Image.new('RGB', (height, height))

    if pred is not None:
        pred = Im(pred).pil.resize((height, height))

    dst = Image.new('RGBA', (width, height))
    
    if bev:
        bev_width, bev_height = bev.size[0], bev.size[1]
        if add_car:
            bev = ImageDraw.Draw(bev)
            width_, height_ = 6/256 * bev_width, 12/256 * bev_height
            bev.rectangle((bev_width // 2 - width_, bev_height // 2 - height_, bev_width // 2 + width_, bev_height // 2 + height_), fill="#00FF11")
            bev = bev._image
        
        dst.paste(bev, (5 + landscape_width, landscape_height + vert_padding))
    else:
        bev_width = 0

    bev_width = 0

    if pred:
        dst.paste(pred, (horiz_padding + bev_width, 0))
        pred_width = pred.size[0] + horiz_padding

    add_num = (landscape_height * 2 - landscape_width) // 2
    dst.paste(images['CAM_FRONT_LEFT'], (bev_width + horiz_padding, landscape_height - add_num))
    dst.paste(images['CAM_FRONT'], (5 + bev_width + landscape_width + 2 * horiz_padding, 0))
    dst.paste(images['CAM_FRONT_RIGHT'], (10 + bev_width + 2 * landscape_width + 3 * horiz_padding, landscape_height - add_num))

    if six_cameras:
        dst.paste(images['CAM_BACK_LEFT'].transpose(Image.Transpose.FLIP_LEFT_RIGHT), (bev_width + horiz_padding, landscape_height + vert_padding + landscape_height - add_num))
        dst.paste(images['CAM_BACK'].transpose(Image.Transpose.FLIP_LEFT_RIGHT), (5 + bev_width + landscape_width + 2 * horiz_padding, 2 * vert_padding + landscape_height + landscape_height + landscape_height - (2 * add_num)))
        dst.paste(images['CAM_BACK_RIGHT'].transpose(Image.Transpose.FLIP_LEFT_RIGHT), (10 + bev_width + 2 * landscape_width + 3 * horiz_padding, landscape_height + vert_padding + landscape_height - add_num))

    return dst

def camera_bev_grid(images: dict, bev=None, pred=None, keep_bev_space=False, add_car=True):
    images = {k: (Im(v).pil if not isinstance(v, Image.Image) else v) for k, v in images.items()}

    landscape_width = images[next(iter(images))].size[0]
    landscape_height = images[next(iter(images))].size[1]

    horiz_padding = 5
    vert_padding = 5

    six_cameras = len(images) == 6
    height = landscape_height + vert_padding + (landscape_height if six_cameras else 0)
    width = landscape_width + landscape_width + landscape_width + (landscape_height * 2) + 3 * horiz_padding + (landscape_height * 2 + horiz_padding if pred is not None else 0)
    pred_width = 0

    if bev is not None:
        bev = viz_bev(bev, dataset=Dataset.NUSCENES).pil.resize((height, height))
    elif keep_bev_space:
        bev = Image.new('RGB', (height, height))

    if pred is not None:
        pred = Im(pred).pil.resize((height, height))

    dst = Image.new('RGBA', (width, height))

    if bev:
        bev_width, bev_height = bev.size[0], bev.size[1]
        if add_car:
            bev = ImageDraw.Draw(bev)
            bev.rectangle((bev_width // 2 - 6, bev_height // 2 - 12, bev_width // 2 + 6, bev_height // 2 + 12), fill="#00FF11")
            bev = bev._image
        dst.paste(bev, (0, 0))
    else:
        bev_width = 0

    if pred:
        dst.paste(pred, (horiz_padding + bev_width, 0))
        pred_width = pred.size[0] + horiz_padding

    dst.paste(images['CAM_FRONT_LEFT'], (pred_width + bev_width + horiz_padding, 0))
    dst.paste(images['CAM_FRONT'], (pred_width + bev_width + landscape_width + 2 * horiz_padding, 0))
    dst.paste(images['CAM_FRONT_RIGHT'], (pred_width + bev_width + 2 * landscape_width + 3 * horiz_padding, 0))

    if six_cameras:
        dst.paste(images['CAM_BACK_LEFT'].transpose(Image.Transpose.FLIP_LEFT_RIGHT), (pred_width + bev_width + horiz_padding, landscape_height + vert_padding))
        dst.paste(images['CAM_BACK'].transpose(Image.Transpose.FLIP_LEFT_RIGHT), (pred_width + bev_width + landscape_width + 2 * horiz_padding, landscape_height + vert_padding))
        dst.paste(images['CAM_BACK_RIGHT'].transpose(Image.Transpose.FLIP_LEFT_RIGHT), (pred_width + bev_width + 2 * landscape_width + 3 * horiz_padding, landscape_height + vert_padding))

    return dst


def argoverse_camera_bev_grid(images: dict, bev=None, keep_bev_space=False, add_car=True):
    images = {k: (Im(v).pil if not isinstance(v, Image.Image) else v) for k, v in images.items()}

    landscape_width = images[next(iter(images))].size[0]
    landscape_height = images[next(iter(images))].size[1]

    horiz_padding = 5

    height = landscape_height
    width = len(images) * landscape_width + (landscape_height) + 4 * horiz_padding

    if bev is not None:
        bev = viz_bev(bev, dataset=Dataset.ARGOVERSE).pil.resize((height, height))
    elif keep_bev_space:
        bev = Image.new('RGB', (height, height))

    dst = Image.new('RGBA', (width, height))

    if bev:
        bev_width, bev_height = bev.size[0], bev.size[1]
        if add_car:
            bev = ImageDraw.Draw(bev)
            bev.rectangle((bev_width // 2 - 4, bev_height // 2 - 8, bev_width // 2 + 4, bev_height // 2 + 8), fill="#00FF11")
            bev = bev._image
        dst.paste(bev, (0, 0))
    else:
        bev_width = 0

    if len(images) == 4:
        dst.paste(images['ring_side_left'], (bev_width + horiz_padding, 0))
        dst.paste(images['ring_front_left'], (bev_width + 1 * landscape_width + 2 * horiz_padding, 0))
        dst.paste(images['ring_front_right'], (bev_width + 2 * landscape_width + 3 * horiz_padding, 0))
        dst.paste(images['ring_side_right'], (bev_width + 3 * landscape_width + 4 * horiz_padding, 0))
    elif len(images) == 1:
        dst.paste(next(iter(images.values())), (bev_width + horiz_padding, 0))
    elif len(images) == 3:
        dst.paste(images['ring_front_left'], (bev_width + horiz_padding, 0))
        dst.paste(images['ring_front_center'], (bev_width + 1 * landscape_height + 2 * horiz_padding, 0))
        dst.paste(images['ring_front_right'], (bev_width + 2 * landscape_height + 3 * horiz_padding, 0))

    return dst


if __name__ == "__main__":
    images = {k: Im(torch.randn((224, 400, 3))).pil for k in Cameras.NUSCENES_ABLATION_CAMERAS}
    camera_bev_grid(images, torch.randn((256, 256, 21)))
    batched_camera_bev_grid(torch.randn((2, 6, 256, 256, 3)))
