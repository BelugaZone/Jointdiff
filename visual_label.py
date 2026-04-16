import os
import sys
sys.path.append(os.getcwd())
import warnings 
warnings.filterwarnings("ignore")
import wandb
from tqdm import tqdm
import torch
import time
import random
import numpy as np
from torchvision.transforms import ToPILImage
from typing import Union
from argparse import ArgumentParser
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import accelerate
from multi_view_generation.bev_utils import viz_bev , visualize_map_mask
from multi_view_generation.bev_utils import bev_pixels_to_cam , render_map_in_image, cam_pixels_to_bev
from multi_view_generation.bev_utils import MapConverter
from multi_view_generation.bev_utils.nuscenes_dataset import NuScenesDataset
import json
from copy import copy
from diffusers.utils import make_image_grid
import gradio as gr
from patch import patch
from pipeline.pipeline_unicon import StableDiffusionUniConPipeline
from utils.utils import blip2_cap, get_combined_filename, save_png_with_comment, set_unicon_config_inference, parse_schedule, process_images
from utils.load_utils import load_model_configs, load_unicon, load_blip_processor, load_annotator, annotator_dict, load_unicon_weights, load_scheduler
from multi_view_generation.bev_utils import viz_bev
from multi_view_generation.bev_utils import bev_pixels_to_cam , render_map_in_image, cam_pixels_to_bev
from utils.utils import get_prompt
from skimage.transform import resize
import glob
from accelerate import DistributedDataParallelKwargs
from utils.confusion import BinaryConfusionMatrix

def encode_binary_labels(masks):
    bits = np.power(2, np.arange(len(masks), dtype=np.int32))
    return (masks.astype(np.int32) * bits.reshape(-1, 1, 1)).sum(0)

def decode_binary_labels(labels, nclass):
    bits = torch.pow(2, torch.arange(nclass))
    return (labels & bits.view(-1, 1, 1)) > 0

def process_segmentation(y_img_pil):
            E01 = np.array([
               [ 5.6847786e-03, -9.9998349e-01,  8.0507132e-04,  5.0603189e-03],
               [-5.6366678e-03, -8.3711528e-04, -9.9998379e-01,  1.5205332e+00],
               [ 9.9996793e-01,  5.6801485e-03, -5.6413338e-03, -1.6923035e+00],
               [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
                ])
            
            intrinsics01 = np.array([
                       [350.7877, 0.0, 231.4447],
                       [0.0, 356.3557, 133.6845],
                       [0.0, 0.0, 1.0]
                       ])
            # intrinsics01 = np.array([
            #            [405.26,   0.0,         261.2],
            #            [  0.0,         720.44, 279.62],
            #            [  0.0,           0.0,           1.0        ]
            #            ])
            print('y_img_pil',y_img_pil.shape)
            if y_img_pil.shape[2] == 6:
                 y_tensor = ((y_img_pil + 1) / 2).clip(0, 1).cpu().numpy()
            else:
                 y_tensor = ((y_img_pil + 1) / 2).clip(0, 1).permute(1,2,0).cpu().numpy()
            
            y_tensor = (y_tensor > 0.5)
            y_tensor_flip = np.flipud(y_tensor)

            # 保存为可用数据
            label_tensor = y_tensor.transpose(2,0,1)
            labels = encode_binary_labels(label_tensor)

            # Save outputs to disk
            


            channel_coords = bev_pixels_to_cam(y_tensor_flip, E01)
            bev_cam = render_map_in_image(intrinsics01, channel_coords)
            bev_cam_imge = viz_bev(bev_cam)
            colored_seg = viz_bev(y_tensor_flip)  # 假设使用 nuScenes 数据集
            return colored_seg.pil , bev_cam_imge.pil ,labels

E01 = np.array([
               [ 5.6847786e-03, -9.9998349e-01,  8.0507132e-04,  5.0603189e-03],
               [-5.6366678e-03, -8.3711528e-04, -9.9998379e-01,  1.5205332e+00],
               [ 9.9996793e-01,  5.6801485e-03, -5.6413338e-03, -1.6923035e+00],
               [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
                ])
            
intrinsics01 = np.array([
                       [350.7877, 0.0, 231.4447],
                       [0.0, 356.3557, 133.6845],
                       [0.0, 0.0, 1.0]
                       ])

# 处理BEV标签
bev_dir = '/opt/data/private/hwj_autodrive/UniCon/AAstanderd_imggen/448/imggen_finalmodel_448_45000:21.83/label'
output_dir_origin = '/opt/data/private/hwj_autodrive/UniCon/AAstanderd_imggen/448/imggen_finalmodel_448_45000:21.83/bev_4c'
output_dir_project = '/opt/data/private/hwj_autodrive/UniCon/AAstanderd_imggen/448/imggen_finalmodel_448_45000:21.83/project_4c'

# 确保输出目录存在
os.makedirs(output_dir_origin, exist_ok=True)
os.makedirs(output_dir_project, exist_ok=True)

# 获取所有BEV图片路径
bev_paths = sorted(glob.glob(os.path.join(bev_dir, '*.png')))  # 根据实际格式调整

for idx, bev_path in enumerate(bev_paths):
    # 处理单张图片
#     encoded_labels = torch.tensor(np.array(Image.open(bev_path))).long()
     img_array = np.array(Image.open(bev_path))
     # 先转换为 int32 或 int64（PyTorch 支持的类型）
     img_array = img_array.astype(np.int32)  # 或者 np.int64
     encoded_labels = torch.tensor(img_array).long()

     binary_labels = decode_binary_labels(encoded_labels, 6)
     
     bev_tensor = binary_labels[:4, :, :]  # [C, H, W]
     bev_tensor = (bev_tensor > 0.5).float()
     bev_tensor = bev_tensor.permute(1, 2, 0)
     bev_tensor_flip = np.flipud(bev_tensor)
     
     # 可视化原始BEV
     decode_bev = viz_bev(bev_tensor_flip).pil
     
     # 投影处理
     
     channel_coords = bev_pixels_to_cam(bev_tensor_flip, E01)
     bev_cam = render_map_in_image(intrinsics01, channel_coords)
     bev_cam_image = viz_bev(bev_cam).pil
     
     # 保存结果
     filename = os.path.basename(bev_path)
     origin_path = os.path.join(output_dir_origin, f"bev4c_{filename}")
     project_path = os.path.join(output_dir_project, f"project4c_{filename}")
     
     decode_bev.save(origin_path)
     bev_cam_image.save(project_path)
     
     print(f'Processed {idx+1}/{len(bev_paths)}: {filename}')

print('All BEV images processed successfully!')

