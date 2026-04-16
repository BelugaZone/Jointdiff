from image_utils import Im

import argparse, os, sys, datetime, glob, importlib, csv
from torch.utils.data import random_split, DataLoader, Dataset, Subset, default_collate


import os

from typing import Dict, List, Tuple, Optional, Union
from multi_view_generation.bev_utils import render_helper
import cv2
import descartes
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Arrow
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from pyquaternion import Quaternion
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
from tqdm import tqdm
import torch
from nuscenes.map_expansion.arcline_path_utils import discretize_lane, ArcLinePath
from nuscenes.map_expansion.bitmap import BitMap
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from omegaconf import OmegaConf
from scipy.ndimage import binary_dilation

bev = {"h": 256, "w": 256, "h_meters": 50, "w_meters": 50, "offset": 0.51}
bev_shape = (256, 448)


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    sh = h / h_meters
    sw = w / w_meters

    return np.float32([[0.0, -sw, w * offset + w / 2.0], [-sh, 0.0, h / 2.0], [0.0, 0.0, 1.0]])

view = get_view_matrix(**bev)


def _clip_points_behind_camera(points, near_plane: float):
    """
    Perform clipping on polygons that are partially behind the camera.
    This method is necessary as the projection does not work for points behind the camera.
    Hence we compute the line between the point and the camera and follow that line until we hit the near plane of
    the camera. Then we use that point.
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param near_plane: If we set the near_plane distance of the camera to 0 then some points will project to
        infinity. Therefore we need to clip these points at the near plane.
    :return: The clipped version of the polygon. This may have fewer points than the original polygon if some lines
        were entirely behind the polygon.
    """
    points_clipped = []
    # Loop through each line on the polygon.
    # For each line where exactly 1 endpoints is behind the camera, move the point along the line until
    # it hits the near plane of the camera (clipping).
    assert points.shape[0] == 3
    point_count = points.shape[1]
    for line_1 in range(point_count):
        line_2 = (line_1 + 1) % point_count
        point_1 = points[:, line_1]
        point_2 = points[:, line_2]
        z_1 = point_1[2]
        z_2 = point_2[2]

        if z_1 >= near_plane and z_2 >= near_plane:
            # Both points are in front.
            # Add both points unless the first is already added.
            if len(points_clipped) == 0 or all(points_clipped[-1] != point_1):
                points_clipped.append(point_1)
            points_clipped.append(point_2)
        elif z_1 < near_plane and z_2 < near_plane:
            # Both points are in behind.
            # Don't add anything.
            continue
        else:
            # One point is in front, one behind.
            # By convention pointA is behind the camera and pointB in front.
            if z_1 <= z_2:
                point_a = points[:, line_1]
                point_b = points[:, line_2]
            else:
                point_a = points[:, line_2]
                point_b = points[:, line_1]
            z_a = point_a[2]
            z_b = point_b[2]

            # Clip line along near plane.
            pointdiff = point_b - point_a
            alpha = (near_plane - z_b) / (z_a - z_b)
            clipped = point_a + (1 - alpha) * pointdiff
            assert np.abs(clipped[2] - near_plane) < 1e-6

            # Add the first point (if valid and not duplicate), the clipped point and the second point (if valid).
            if z_1 >= near_plane and (len(points_clipped) == 0 or all(points_clipped[-1] != point_1)):
                points_clipped.append(point_1)
            points_clipped.append(clipped)
            if z_2 >= near_plane:
                points_clipped.append(point_2)

    points_clipped = np.array(points_clipped).transpose()
    return points_clipped


def render_map_in_image(
                        intrinsics,
                        channel_coords,
                        camera_channel: str = 'CAM_FRONT',
                        alpha: float = 1.0,
                        min_polygon_area: float = 0,
                        render_behind_cam: bool = True,
                        render_outside_im: bool = True,
                        layer_names: List[str] = None,
                        verbose: bool = True,
                        out_path: str = None) -> Tuple[Figure, Axes]:
    """
    Render a nuScenes camera image and overlay the polygons for the specified map layers.
    Note that the projections are not always accurate as the localization is in 2d.
    :param nusc: The NuScenes instance to load the image from.
    :param sample_token: The image's corresponding sample_token.
    :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
    :param alpha: The transparency value of the layers to render in [0, 1].
    :param patch_radius: The radius in meters around the ego car in which to select map records.
    :param min_polygon_area: Minimum area a polygon needs to have to be rendered.
    :param render_behind_cam: Whether to render polygons where any point is behind the camera.
    :param render_outside_im: Whether to render polygons where any point is outside the image.
    :param layer_names: The names of the layers to render, e.g. ['lane'].
        If set to None, the recommended setting will be used.
    :param verbose: Whether to print to stdout.
    :param out_path: Optional path to save the rendered figure to disk.
    """
    near_plane = 1e-8

    color_map = dict(
              drivable_area = "#00ff00",  # green
              ped_crossing = "#ff69b4",  # hotpink
              walkway =  "#ffd700",  # gold
              carpark_area  = "#ffa500",  # orange
              bus =  "#ff7f50",  # coral
              bicycle =  "#dc143c",  # crimson
              car =  "#ff9e00",  # orange
              construction_vehicle =  "#e99646",  # darkorange
              motorcycle =  "#ff3d63",  # deep pink
              trailer =  "#ff8c00",  # darkorange
              truck =  "#ff6347",  # tomato
              pedestrian =  "#0000e6",  # blue
              traffic_cone =  "#ffa500",  # orange
              barrier = "#696969",  # dimgray
              nothing =  "#c8c8c8"  # gray
    )

    if verbose:
        print('Warning: Note that the projections are not always accurate as the localization is in 2d.')

    # Default layers.
    if layer_names is None:
        layer_names = ['drivable_area', 'ped_crossing', 'walkway', 'carpark_area', 'bus',
                       'bicycle', 'car', 'construction_vehicle', 'motorcycle', 'trailer', 'truck', 'pedestrian'
                       ]
        # layer_names = ['drivable_area', 'ped_crossing', 'walkway'
        #                ]

    # Check layers whether we can render them.

    # Check that NuScenesMap was loaded for the correct location.
    im_size = (bev_shape[0],bev_shape[1])
    cam_intrinsic = intrinsics
    multi_channel_image = np.zeros((bev_shape[0], bev_shape[1], len(layer_names)), dtype=np.uint8)
 
    for c, (layer_name, points_coords) in enumerate(zip(layer_names, channel_coords)):  # 步骤3：配对图层和通道数据  # 步骤3：配对图层和通道数据
        points = np.array(points_coords).T

        depths = points[2, :]
        behind = depths < near_plane

        # 投影时不归一化（保留深度信息）
        points_proj = view_points(points, cam_intrinsic, normalize=False)


        # 手动计算归一化坐标（避免反向投影时丢失深度）
        x_norm = points_proj[0, :] / points_proj[2, :]
        y_norm = points_proj[1, :] / points_proj[2, :]

     
        x_pixel = np.round(x_norm).astype(int)
        y_pixel = np.round(y_norm).astype(int)

        # 筛选有效坐标
        valid = (x_pixel >= 0) & (x_pixel < bev_shape[1]) & \
                (y_pixel >= 0) & (y_pixel < bev_shape[0])
        x_valid = x_pixel[valid]
        y_valid = y_pixel[valid]

        # 生成二值图
        binary_image = np.zeros((bev_shape[0], bev_shape[1]), dtype=np.uint8)
        binary_image[y_valid, x_valid] = 1
        struct_elem = np.ones((1, 1), dtype=bool)  # 3x3结构元素
        dilated_image = binary_dilation(binary_image, structure=struct_elem)
        binary_image = dilated_image.astype(np.uint8)
        multi_channel_image[:, :, c] = binary_image
    return multi_channel_image


def bev_pixels_to_cam(input_bev, E):
    H, W, C = input_bev.shape
    all_cam_coords = []
    # 二值图在插值到256后会出现非二值点，这也是为什么visual_bev中使用概率填充图像
    threshold = 0.5
    
    # input_bev = (input_bev >= threshold).astype(np.uint8)

    # 预计算逆矩阵（提升效率）
    V_inv = np.linalg.inv(view)
    #M = np.linalg.inv(pose_inverse)  # M = pose矩阵

    for c in range(C):
        mask = input_bev[:, :, c] != 0
        nonzero_indices = np.argwhere(mask)

        if len(nonzero_indices) == 0:
            all_cam_coords.append(np.empty((0, 3)))  # 新增：空通道处理
            continue

        us = nonzero_indices[:, 0].astype(float)
        vs = nonzero_indices[:, 1].astype(float)
        

        u_norm = us / (W - 1)
        v_norm = vs / (H - 1)
        #z_norm = np.zeros_like(u_norm)

        # 1. 构造像素齐次坐标
        pixels_homogeneous = np.column_stack((us, vs, np.ones_like(us)))

        # pixels_homogeneous = np.column_stack((u_norm, v_norm, np.ones_like(us)))
        
        local_coords = (V_inv @ pixels_homogeneous.T).T  # (N, 3)

        # 3. 归一化齐次坐标
        x_local = local_coords[:, 1] / local_coords[:, 2]
        y_local = local_coords[:, 0] / local_coords[:, 2]

        # 4. 构造四维齐次坐标 (egolidarflat坐标系)
        local_4d = np.column_stack((
            x_local,
            y_local,
            np.zeros_like(x_local),
            np.ones_like(x_local)
            ))
      
        cam_4d = (E @ local_4d.T).T  # (N, 4)
        cam_coords = cam_4d[:, :3]  # 提取3D坐标 (x, y, z)
        all_cam_coords.append(cam_coords)  # 存储相机坐标
        # print("all_cam_coords", all_cam_coords.shape)


    return all_cam_coords


def cam_pixels_to_bev(
        cam_image,       # 输入的多通道二值图，shape (H, W, C)
        intrinsics,      # 相机内参矩阵 (3x3)
        E              # 外参矩阵（自车到相机的变换，4x4）
        ):

    # 初始化BEV多通道图像
    bev_shape = (256, 256)
    H, W, C = cam_image.shape
    multi_bev = np.zeros((bev_shape[0], bev_shape[1], C), dtype=np.uint8)

    # 提取自车到相机的旋转和平移
    R_ego_to_cam = E[:3, :3]
    t_ego_to_cam = E[:3, 3]

    # 构造单应性矩阵H：地平面(X,Y,0) -> 图像(u,v)
    H_cam = np.hstack((R_ego_to_cam[:, :2], t_ego_to_cam.reshape(-1, 1)))
    H = intrinsics @ H_cam
    H_inv = np.linalg.inv(H)  # 图像(u,v) -> 地平面(X,Y)

    # 遍历每个通道的非零像素
    for c in range(C):
        channel_img = cam_image[:, :, c]
        nonzero_indices = np.argwhere(channel_img != 0)
        
        for v, u in nonzero_indices:
            # 图像坐标转地平面坐标
            point_img = np.array([u, v, 1.0])
            point_ground_homo = H_inv @ point_img
            if point_ground_homo[2] == 0:
                continue  # 避免除零
            X_ego, Y_ego = (point_ground_homo[:2] / point_ground_homo[2])

            # 地平面坐标转BEV像素坐标
            point_bev_homo = view @ np.array([X_ego, Y_ego, 1.0])
            bev_u, bev_v = (point_bev_homo[:2] / point_bev_homo[2]).astype(int)

            # 确保坐标在图像范围内
            if 0 <= bev_u < bev_shape[1] and 0 <= bev_v < bev_shape[0]:
                multi_bev[bev_v, bev_u, c] = 1

    return multi_bev


# def cam_pixels_to_bev(binary_image, intrinsics, E, bev_height=256, bev_width=256, 
#                    h_meters=80.0, w_meters=80.0, offset=0.0):
#     """
#     将多通道二值图从摄像机坐标系投影到BEV平面
    
#     参数:
#         binary_image: 输入二值图 [H, W, C]
#         E: 外参矩阵 (4x4)
#         intrinsics: 内参矩阵 (3x3)
#         bev_height: BEV图像高度
#         bev_width: BEV图像宽度
#         h_meters: BEV纵向范围(米)
#         w_meters: BEV横向范围(米)
#         offset: BEV原点偏移系数
        
#     返回:
#         bev_image: BEV投影结果 [bev_height, bev_width, C]
#     """
#     # 生成视图矩阵
    
    
#     # 计算逆矩阵
#     K_inv = np.linalg.inv(intrinsics)  # 内参逆矩阵
#     E_inv = np.linalg.inv(E)           # 外参逆矩阵
    
#     # 分解外参矩阵
#     R = E_inv[:3, :3]   # 旋转分量
#     t = E_inv[:3, 3]    # 平移分量
    
#     # 初始化BEV图像
#     bev_image = np.zeros((bev_height, bev_width, binary_image.shape[2]), dtype=np.uint8)
    
#     # 遍历所有像素
#     for c in range(binary_image.shape[2]):  # 多通道处理
#         for v in range(binary_image.shape[0]):
#             for u in range(binary_image.shape[1]):
#                 if binary_image[v, u, c] == 1:
#                     # 1. 归一化相机坐标
#                     uv_homogeneous = np.array([u, v, 1.0])
#                     norm_coords = K_inv @ uv_homogeneous
#                     x_norm, y_norm, _ = norm_coords / norm_coords[2]
                    
#                     # 2. 计算射线方向
#                     direction_cam = np.array([x_norm, y_norm, 1.0])
#                     direction_world = R @ direction_cam  # 转换到世界坐标系
                    
#                     # 3. 计算与地面交点
#                     denominator = direction_world[2]
#                     if abs(denominator) < 1e-6: continue  # 避免除零错误
                    
#                     lambda_ = -t[2] / denominator
#                     X_world = R[0,0]*x_norm*lambda_ + R[0,1]*y_norm*lambda_ + R[0,2]*lambda_ - (R[0,:3] @ t[:3])
#                     Y_world = R[1,0]*x_norm*lambda_ + R[1,1]*y_norm*lambda_ + R[1,2]*lambda_ - (R[1,:3] @ t[:3])
                    
#                     # 4. BEV坐标映射
#                     bev_coords = view @ np.array([X_world, Y_world, 1.0])
#                     u_bev = int(bev_coords[0] + 0.5)  # 四舍五入
#                     v_bev = int(bev_coords[1] + 0.5)
                    
#                     # 5. 写入BEV图像
#                     if 0 <= u_bev < bev_width and 0 <= v_bev < bev_height:
#                         bev_image[v_bev, u_bev, c] = 1
                        
#     return bev_image


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--fid", action="store_true", help="if sampling images for FID"
    )
    ### extra args for distributed
    parser.add_argument(
        "--distributed", action="store_true", help="if using distributed training"
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="total number of nodes",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="total number of GPUs across all nodes, needs to be set if number of GPUs is not equal across all nodes",
    )
    parser.add_argument(
        "--node-id", type=int, default=0, help="ID of the current node [0...world-size]"
    )
    parser.add_argument(
        "--nodes-info",
        type=str,
        default=None,
        help="list of IP addresses for each node",
    )
    return parser


if __name__ == "__main__":
    print("nothing")   