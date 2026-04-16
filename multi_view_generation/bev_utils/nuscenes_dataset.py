import pyrootutils
from PIL import Image, ImageDraw
from heapq import nlargest
from nuscenes.utils.geometry_utils import view_points
from einops import rearrange
import time
from collections.abc import Iterable
from pathlib import Path
import hashlib
from typing import Dict, Union, Tuple, Optional
import albumentations as A
import cv2
import numpy as np
import torch
import pickle
from pyquaternion import Quaternion
from torch.utils.data import DataLoader, Dataset
from os.path import exists
from torchvision.transforms.functional import to_tensor
import torchvision.transforms.functional as F
import torchvision.transforms as T
from multi_view_generation.bev_utils.nuscenes_helper import polar_transform
import os
from numpy.linalg import inv
from tqdm import tqdm
from nuscenes.utils.geometry_utils import BoxVisibility
from multi_view_generation.bev_utils import Cameras
from multi_view_generation.bev_utils import util, raw_output_data_bev_grid, NUSCENES_DIR, SAVE_DATA_DIR
from multi_view_generation.bev_utils import render_helper
from multi_view_generation.bev_utils import bev_pixels_to_cam , render_map_in_image
from multi_view_generation.bev_utils.nuscenes_helper import (
    Split,
    NuScenesSingleton,
    get_split,
    compute_pixel_ray_directions,
    quaternion_yaw,
    get_annotations_by_category,
    get_dynamic_layers,
    get_line_layers,
    get_static_layers,
    parse_pose,
    parse_scene,
    decode_binary_labels,
    decode,
    encode,
    split_range,
    get_dynamic_objects,
    DIVIDER,
    STATIC,
    DYNAMIC,
    NUM_CLASSES,
    NusceneCamGeometry,
)
import random

from multi_view_generation.bev_utils import viz_bev

class NuScenesDataset(Dataset):
    def __init__(
        self,
        split: Split,
        dataset_dir: Path = NUSCENES_DIR,  # v1.0-mini and/or v1.0-trainval should be inside
        return_cam_img: bool = True,
        return_bev_img: bool = False,
        mini_dataset: bool = False,  # Mini dataset loads much faster for debugging
        cam_res: Tuple[int, int] = (256, 256),  # Downscale camera image to this (both H,W)
        bev_res: int = 96,  # Downscale BEV to this (both H,W)
        augment_cam_img: bool = False,
        augment_bev_img: bool = False,
        normalize_cam_img: bool = True,  # Normalize all camera images
        flip_imgs: bool = False,
        stage_2_training: bool = False,  # Returns images with a smaller FOV (square crop)
        metadrive_compatible: bool = False,  # Returns 3-channel BEV image compatible with metadrive output
        metadrive_compatible_v2: bool = False,  # Returns 3-channel BEV image compatible with metadrive output
        metadrive_only: bool = False,
        fov_bev_only: bool = False,  # Returns BEV limited to FOV of camera
        only_keyframes: bool = True,
        return_metadrive_bev: bool = False,  # Randomly returns Metadrive generated BEV images half the time
        return_all_cams: bool = False,
        eval_generate: Union[bool, str] = False,
        single_camera_only: bool = False,  # Returns images from a single camera (i.e. only front camera). Returns "image_paths" that reference the other 5 camera images
        non_square_images: bool = False,
        generate_split: Union[bool, Tuple[int, int, int]] = False,
        select_cameras: Optional[Tuple] = None,
        specific_metadrive_bev: Optional[str] = None,
        specific_nuscenes_token: Optional[str] = None,
        frozen_image_tokens: Optional[str] = None,
        frozen_scene_tokens: Optional[str] = None,
        same_bev_rand_density: Optional[int] = None,
        duplicate_tokens_generation: Optional[int] = None,
        **kwargs,
    ):
        # Hack to remove boilerplate self.x = x code
        
        for k, v in vars().items():
            if k != 'self':
                setattr(self, k, v)

        self.kwargs = kwargs
        self.images_data = None
        self.mini_dataset = mini_dataset
        self.single_camera_only = single_camera_only
        self.return_all_cams = return_all_cams
        self.only_keyframes = only_keyframes


        self.split = Split(split)
        if not isinstance(dataset_dir, Path):
            self.dataset_dir = Path(dataset_dir)

        self.select_cameras = self.select_cameras if self.select_cameras else [0]
        self.cameras = [[0, 1, 2, 3, 4, 5]]
        self.CAMERAS = [0]

        # self.cam_res = cam_res if isinstance(cam_res, Iterable) else (cam_res, cam_res)
        self.cam_res = cam_res

        self.split_scenes = get_split(self.split.name.lower())
        if generate_split:
            import itertools
            self.split_scenes = split_range(self.split_scenes, generate_split[0])
            self.split_scenes = next(itertools.islice(self.split_scenes, generate_split[1], generate_split[1] + 1))

        helper = None

        dataset_type = "v1.0-mini" if self.mini_dataset else "v1.0-trainval"
        pickle_name = Path.home() / ".cache" / "nuscenes" / f"nusc-{dataset_type}.p"

        if exists(pickle_name):
            start_time = time.time()
            with open(pickle_name, "rb") as f:
                helper = pickle.load(f)
            print(f'Took {round(time.time() - start_time, 1)} seconds to load NuScenes {dataset_type}')
        else:
            pickle_name.parent.mkdir(parents=True, exist_ok=True)
            helper = NuScenesSingleton(dataset_dir, dataset_type)
            with open(pickle_name, "wb") as f:  # "wb" because we want to write in binary mode
                pickle.dump(helper, f)

        self.nusc = helper.nusc

        pickle_name = Path.home() / ".cache" / "nuscenes" / f"nusc-{dataset_type}-{self.split.name.lower()}.p"
        if exists(pickle_name):
            start_time = time.time()
            with open(pickle_name, "rb") as f:
                self.samples, self.nusc_map = pickle.load(f)
            print(f'Took {round(time.time() - start_time, 1)} seconds to load NuScenes {dataset_type}-{self.split.name.lower()}')
        else:
            self.samples = []
            self.nusc_map = {}
            for scene_name, scene_record in helper.get_scenes():
                if scene_name not in self.split_scenes:
                    continue

                self.samples.extend(parse_scene(self.nusc, scene_record, self.cameras))
                self.nusc_map[scene_name] = helper.get_map(scene_record["log_token"])

            if not generate_split:
                with open(pickle_name, "wb") as f:  # "wb" because we want to write in binary mode
                    pickle.dump((self.samples, self.nusc_map), f)

        # self.cam_transform = A.Compose([])

        self.max_h_pre_crop = None
        self.cam_intrinsic_aug = NusceneCamGeometry()

        # if self.augment_cam_img:
        #     self.cam_transform = A.Compose([A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5)])
        #     if not self.non_square_images:
        #         self.max_h_pre_crop = int(1.125 * max(self.cam_res))
        #         self.cam_transform = A.Compose([self.cam_transform, A.RandomCrop(*self.cam_res)])
        # elif not self.non_square_images:
        #     self.max_h_pre_crop = cam_res
        #     self.cam_transform = A.Compose(
        #         [
        #             A.CenterCrop(*self.cam_res),
        #         ]
        #     )

        # if self.flip_imgs:
        #     self.cam_transform = A.Compose([self.cam_transform, A.HorizontalFlip(p=0.5)])

        # if normalize_cam_img and not self.stage_2_training:
        #     self.cam_transform = A.Compose(
        #         [
        #             self.cam_transform,
        #             A.Normalize(mean=(0.4265, 0.4489, 0.4769), std=(0.2053, 0.2206, 0.2578), max_pixel_value=255.0),
        #         ]
        #     )

        # # Override all previously set augs
        # if self.non_square_images:
        #     if self.stage_2_training:
        #         self.color_transform = T.ColorJitter(brightness=0.1, contrast=0.05, saturation=0.05, hue=0.05)
        #     else:
        #         self.color_transform = T.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1, hue=0.1)

        #     self.cam_transform = T.Compose([
        #         T.Normalize(
        #             mean=(0.4265, 0.4489, 0.4769),
        #             std=(0.2053, 0.2206, 0.2578),
        #         ),
        #     ])

        # if "square_images_cityscapes" in kwargs:
        #     self.cam_transform = T.Compose(
        #         [
        #             T.CenterCrop(900), T.Resize((256, 256), T.InterpolationMode.BICUBIC), self.cam_transform
        #         ]
        #     )

        # self.bev_transform = A.Compose([A.Resize(bev_res, bev_res, interpolation=cv2.INTER_LANCZOS4)])

        # if self.stage_2_training:
        #     if augment_bev_img and self.split == Split.TRAIN:
        #         self.bev_transform = A.Compose([A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.01, rotate_limit=1, p=0.5)])
        #     else:
        #         self.bev_transform = A.Compose([])
        # elif self.metadrive_compatible_v2:
        #     if augment_bev_img:
        #         self.bev_transform = A.Compose([A.ShiftScaleRotate(shift_limit=0.075, scale_limit=0.075, rotate_limit=10, p=0.5), A.HorizontalFlip(p=0.5)])
        #     else:
        #         self.bev_transform = A.Compose([])

        sha = hashlib.sha1(f"{','.join([*self.split_scenes, dataset_type, str(only_keyframes)])}".encode("utf-8"))
        pickle_name = Path.home() / ".cache" / "nuscenes" / f"{sha.hexdigest()}.p"
        
        self.images_data = [
                s["token"]
                for s in self.nusc.sample_data
                if (
                        s["sensor_modality"] == "camera"
                        and self.nusc.get("scene", self.nusc.get("sample", s["sample_token"])["scene_token"])[
                            "name"] in self.split_scenes
                        and s["channel"] == "CAM_FRONT"
                        and (s['is_key_frame'])
                )
            ]
        print("original images length===",len(self.images_data))

        if not only_keyframes:
            if exists(pickle_name):
                with open(pickle_name, "rb") as f:
                    self.images = pickle.load(f)

                
            else:
                cam_names = [Cameras.NUSCENES_CAMERAS[cam_idx] for cam_idx in self.select_cameras]
                self.images = []
                total_skipped = 0

                scenes = helper.get_scenes()

                for scene_name, scene_record in tqdm(scenes):
                    if scene_name in self.split_scenes:
                        next_lidar_sample_token = self.nusc.get("sample", scene_record["first_sample_token"])['data']['LIDAR_TOP']
                        while next_lidar_sample_token != '':
                            lidar_sample = self.nusc.get("sample_data", next_lidar_sample_token)
                            next_lidar_sample_token = lidar_sample['next']
                            lidar_sample_timestamp = lidar_sample['timestamp']
                            right_sample = self.nusc.get("sample", lidar_sample['sample_token'])

                            paired_cam_tokens = {'LIDAR_TOP': lidar_sample['token']}
                            skip_sample = False
                            for cam_name in cam_names:
                                if lidar_sample['is_key_frame']:
                                    paired_cam_tokens[cam_name] = right_sample['data'][cam_name]
                                else:
                                    cur_sample = self.nusc.get("sample_data", right_sample['data'][cam_name])
                                    closest = [None, np.inf]
                                    while True:
                                        if (dist:=np.abs(lidar_sample_timestamp - cur_sample['timestamp'])) < closest[1]:
                                            closest = (cur_sample['token'], dist)
                                        cur_sample = self.nusc.get("sample_data", cur_sample['prev'])
                                        if lidar_sample_timestamp > cur_sample['timestamp']:
                                            break
                                    if closest[1] > 100 * 1e3:
                                        print(f'Warning: {closest[1] * 1e-3} ms between lidar and {cam_name} sample, skipping')
                                        skip_sample = True
                                        break
                                    paired_cam_tokens[cam_name] = closest[0]
                            if not skip_sample:
                                self.images.append(paired_cam_tokens)
                            else:
                                total_skipped += 1
                print(len(self.images), total_skipped)
                with open(pickle_name, "wb") as f:
                    pickle.dump(self.images, f)
        elif self.single_camera_only or self.return_all_cams:
            print("self.single_camera_only or self.return_all_cams=====")
            if self.frozen_image_tokens and exists(self.frozen_image_tokens):
                with open(self.frozen_image_tokens) as f:
                    self.images = [line.rstrip() for line in f]
                print(f'Loaded from {self.frozen_image_tokens} with {len(self.images)} instances')
            else:
                if self.frozen_scene_tokens and exists(self.frozen_scene_tokens):
                    with open(self.frozen_scene_tokens) as f:
                        self.split_scenes = [line.rstrip() for line in f]

                    split_scenes_ = []
                    for scene_token in self.split_scenes:
                        try:
                            split_scenes_.append(self.nusc.get("scene", scene_token)["name"])
                        except:
                            continue
                    self.split_scenes = split_scenes_
                    print(f'Loaded from {self.frozen_scene_tokens} with {len(self.split_scenes)} scenes')
                print("running FRONTcate===============")
                self.images = [
                    s["token"]
                    for s in self.nusc.sample_data
                    if (
                        s["sensor_modality"] == "camera"
                        and self.nusc.get("scene", self.nusc.get("sample", s["sample_token"])["scene_token"])["name"] in self.split_scenes
                        and s["channel"] == "CAM_FRONT"
                        and (s['is_key_frame'] or not only_keyframes)
                    )
                ]

                if generate_split:
                    self.images = self.images[::generate_split[2]]

                if self.frozen_image_tokens:
                    with open(self.frozen_image_tokens, 'w') as f:
                        f.write('\n'.join(self.images))

            if eval_generate:
                initial_size = len(self.images)

                def filter_samples(s):
                    sample_data = self.nusc.get("sample", self.nusc.get('sample_data', s)["sample_token"])["data"]
                    for cam_idx in self.select_cameras:
                        img_path = Path(eval_generate) / 'gen' / self.nusc.get("sample_data", sample_data[Cameras.NUSCENES_CAMERAS[cam_idx]])["filename"]
                        if not exists(img_path):
                            return True
                    return False
                self.images = list(filter(filter_samples, self.images))
                print(f'Removed {initial_size - len(self.images)} for a total of {len(self.images)} instances')

            if duplicate_tokens_generation:
                # This allows us to generate multiple times for a single instance
                self.images = list(np.repeat(self.images, duplicate_tokens_generation))
        else:
            if exists(pickle_name):
                with open(pickle_name, "rb") as f:
                    self.images = pickle.load(f)
            else:
                self.images = [
                    s["token"]
                    for s in self.nusc.sample_data
                    if (s["sensor_modality"] == "camera" and self.nusc.get("scene", self.nusc.get("sample", s["sample_token"])["scene_token"])["name"] in self.split_scenes and (s["is_key_frame"] or not self.only_keyframes))
                ]
                with open(pickle_name, "wb") as f:
                    pickle.dump(self.images, f)

        if self.return_metadrive_bev:
            if not hasattr(NuScenesDataset, "env"):
                from model.multi_view_generation.bev_utils.top_down_obs_nuscenes import TopDownMetaDriveEnvV3
                NuScenesDataset.env = TopDownMetaDriveEnvV3(dict(environment_num=1000, map=10, start_seed=np.random.randint(0, 1000)))
                NuScenesDataset.env.reset()

        if self.same_bev_rand_density is not None:
            self.specific_nuscenes_token = self.images[torch.randint(len(self.images), ()).item()]
            self.same_bev_rand_density_cur = 0.0

        scene_count = len(self.split_scenes)
        print(f'NuScenes has {len(self.images)} images from {scene_count} scenes 'f'on split: {self.split} (dataset: {dataset_type})')

    def reset_selected(self):
        print('Reset selected!')
        self.specific_nuscenes_token = self.images[torch.randint(len(self.images), ()).item()]
        self.same_bev_rand_density_cur = 0.0


    def load_transform_image(self, image_path, cam, color_params=None):
        img = Image.open(self.dataset_dir / image_path)
        img_resize = img.resize((self.cam_res[0],self.cam_res[1]), Image.Resampling.BICUBIC)
        im_to_process = np.asarray(img)
        img_origin = np.asarray(img)
        img_resize = np.asarray(img_resize)
        center_pixel = im_to_process.shape[1] // 2
        self.max_h_pre_crop = self.cam_res[0]

        if self.max_h_pre_crop is not None:
                scale = self.max_h_pre_crop / float(max(im_to_process.shape[1],
                                                        im_to_process.shape[0]))
                self.cam_intrinsic_aug.set_scale(self.cam_res[0]/1600, self.cam_res[1]/900)
                im_to_process = util.smallest_max_size(im_to_process,
                                                        self.max_h_pre_crop)
        
        cam_transform = A.Compose([])
        cam_transform = A.Compose(
                [
                    cam_transform,
                    # A.Normalize(mean=(0.4265, 0.4489, 0.4769), std=(0.2053, 0.2206, 0.2578), max_pixel_value=255.0),
                    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
                    A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0),
                ]
            )
        cam_transform = A.Compose([ cam_transform, A.CenterCrop(height=self.cam_res[1], width=self.cam_res[0])])

        if cam_transform is not None:
                transformed_cam_img = cam_transform(image=img_resize)["image"]
        else:
                transformed_cam_img = im_to_process

        # reform the intrinsics
        camera_intrinsic = self.cam_intrinsic_aug.apply(cam['camera_intrinsic'])
        fx, fy = cam["camera_intrinsic"][0][0], cam["camera_intrinsic"][1][1]
        img_w, img_h = img.size[1], img.size[0]
        center_angle_offset = -compute_pixel_ray_directions(np.array([[center_pixel, img.size[1] / 2]]), fx, fy, img_h, img_w)[0, 0]
        return transformed_cam_img, center_angle_offset, center_pixel, camera_intrinsic

    def get_random_metadrive_bev(self, cam_record_data: Dict):
        """Return BEV image by stepping metadrive sim"""
        bev, r, d, info = NuScenesDataset.env.step([0, 1])
        bev[bev > 0] = 1.0
        if d or np.random.rand() < 0.01:
            NuScenesDataset.env.reset()
        return bev

    def get_metadrive_compatible_bev(self, cam_record_data: Dict):
        """Return BEV image from nuScenes data in a format compatible with Metadrive (3-channel)"""
        to_get = ["car", "truck", "bus", "trailer", "construction", "motorcycle", ]
        anns_dynamic = get_annotations_by_category(self.nusc, cam_record_data, to_get)
        dynamic = get_dynamic_layers(cam_record_data, anns_dynamic)  # 200 200 8
        dynamic = np.logical_or.reduce(dynamic.astype(bool).transpose(2, 0, 1))[..., None]
        dividers = get_line_layers(self.nusc_map, cam_record_data, DIVIDER)  # 200 200 2
        dividers = np.logical_or.reduce(dividers.astype(bool).transpose(2, 0, 1))[..., None]
        static = get_static_layers(self.nusc_map, cam_record_data, STATIC)
        static = np.logical_or.reduce(static.astype(bool).transpose(2, 0, 1))[..., None]
        bev = np.concatenate((static, dividers, dynamic), -1)  # 200 200 12
        bev = (bev / 1.0).astype(np.float32)
        return bev

    def get_metadrive_compatible_bev_v2(self, cam_record_data: Dict):
        """Return BEV image from nuScenes data in 21-channel format"""
        anns_dynamic = get_annotations_by_category(self.nusc, cam_record_data, DYNAMIC)
        anns_dynamic_ = []

        for anns in anns_dynamic:
            anns_dynamic_.append(random.sample(anns, min(int(len(anns) * self.same_bev_rand_density_cur), len(anns))) if self.same_bev_rand_density is not None else anns)

        if hasattr(self, 'same_bev_rand_density_cur') and self.same_bev_rand_density_cur >= 1.0:
            self.reset_selected()

        if self.same_bev_rand_density is not None:
            self.same_bev_rand_density_cur = min(self.same_bev_rand_density + self.same_bev_rand_density_cur, 1)

        anns_vehicle = []
        for anns in anns_dynamic_:
            if len(anns) > 0 and anns[0]['category_name'].startswith('vehicle.'):
                anns_vehicle.extend(anns)

        dynamic = get_dynamic_layers(cam_record_data, anns_dynamic_)  # 200 200 8

        static = get_static_layers(self.nusc_map, cam_record_data, STATIC)  # 200 200 2
        dividers = get_line_layers(self.nusc_map, cam_record_data, DIVIDER)  # 200 200 2

        aux, visibility = get_dynamic_objects(cam_record_data, anns_vehicle)

        # util.get_layered_image_from_binary_mask(static[..., 0][..., None].astype(np.bool)).save('static.png')
        # util.get_layered_image_from_binary_mask(dynamic.astype(np.bool)).save('dynamic.png')
        # util.get_layered_image_from_binary_mask(dividers.astype(np.bool)).save('dividers.png')

        bev = np.concatenate((static, dividers, dynamic, visibility[..., None]), -1)  # 200 200 14
        bev = (bev / 255.0).astype(np.float32)
        bev = np.concatenate((bev, aux), -1)

        return bev

    def get_standard_bev(self, cam_record_data: Dict):
        """Return BEV image (12-channel) that covers 360 deg around ego"""
        # Raw annotations
        anns_dynamic = get_annotations_by_category(self.nusc, cam_record_data, DYNAMIC)
        anns_vehicle = get_annotations_by_category(self.nusc, cam_record_data, ["vehicle"])[0]

        static = get_static_layers(self.nusc_map, cam_record_data, STATIC)  # 200 200 2
        dividers = get_line_layers(self.nusc_map, cam_record_data, DIVIDER)  # 200 200 2
        dynamic = get_dynamic_layers(cam_record_data, anns_dynamic)  # 200 200 8
        bev = np.concatenate((static, dividers, dynamic), -1)  # 200 200 12
        # E01 = np.array([
        #        [ 5.6847786e-03, -9.9998349e-01,  8.0507132e-04,  5.0603189e-03],
        #        [-5.6366678e-03, -8.3711528e-04, -9.9998379e-01,  1.5205332e+00],
        #        [ 9.9996793e-01,  5.6801485e-03, -5.6413338e-03, -1.6923035e+00],
        #        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
        #         ])
            
        # intrinsics01 = np.array([
        #                [202.62675249,   0.0,         130.60272316],
        #                [  0.0,         360.22533776, 139.80645427],
        #                [  0.0,           0.0,           1.0        ]
        #                ])

        # channel_coords = bev_pixels_to_cam(bev, E01)
                      
        # bev_cam = render_map_in_image(intrinsics01, channel_coords)
        # bev = bev_cam         
        #print("dynamic.shape",dynamic.shape)
         # 提取扇形区域
        center=(128, 128)
        start_angle=240
        end_angle=300
        width, height = bev.shape[1], bev.shape[0]
        cx, cy = center

        # 计算覆盖整个图像的最小半径
        corners = [(0, 0), (width-1, 0), (0, height-1), (width-1, height-1)]
        distances = [np.sqrt((cx - x)**2 + (cy - y)**2) for x, y in corners]
        r = int(np.ceil(max(distances)))

    # 生成扇形掩码
        img_mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img_mask)
        draw.pieslice(
           [cx - r, cy - r, cx + r, cy + r],  # 扇形边界框
           start=start_angle, end=end_angle,
           fill=255
        )
        sector_mask = (np.array(img_mask) > 0)  # 转换为布尔掩码
        sector_mask = np.expand_dims(sector_mask, axis=-1)  # 形状 (256, 256, 1)
        sector_mask = np.broadcast_to(sector_mask, (height, width, 13))  # 广播到 (256, 256, 12)

    # 将扇形掩码广播到与原掩码相同的维度并相乘
        bev = bev * sector_mask.astype(bev.dtype)

        bev = (bev / 1.0).astype(np.float32)
        if bev.max() > 1:
           bev = (bev / 255.0)
        

        # hwj modified
        
        bev_resized = cv2.resize(bev, (self.cam_res[0], self.cam_res[1]), interpolation=cv2.INTER_LINEAR)
        bev_resized = 2 * bev_resized - 1 
        # bev_resized = bev_resized[..., [1, 4, 5]]
        bev_resized = bev_resized[..., :12]

        # bev_resized = bev_resized[..., [0, 1, 2, 3, 4, 5, 6]]
        
        # # 生成语义索引（通过 argmax 找激活的通道）
        # semantic_indices = np.argmax(bev_resized, axis=2)  # 形状 [H, W]
        # unique_values = np.unique(semantic_indices)

        # # 打印结果
        # print(f"Unique label values ({len(unique_values)} classes):")
        # print(np.sort(unique_values))

        

        # semantic_indices += 1 
        
        # # 检查像素是否被激活（若所有通道为 False，则未激活）
        # mask_valid = np.any(bev_resized, axis=2)  # 形状 [H, W]
        
        # # 将未激活的像素设为 0
        # semantic_indices = np.where(mask_valid, semantic_indices, 0)

       
        
        # # 将结果存入 labels
        # bev_resized = decode(semantic_indices, 3)

        # bev_resized = 2 * bev_resized - 1

        



        

        return bev_resized

    def get_visible_mask(instrinsics, image_width, extents, resolution):

        # Get calibration parameters
        fu, cu = instrinsics[0, 0], instrinsics[0, 2]

        # Construct a grid of image coordinates
        x1, z1, x2, z2 = extents
        x, z = np.arange(x1, x2, resolution), np.arange(z1, z2, resolution)
        ucoords = x / z[:, None] * fu + cu

        # Return all points which lie within the camera bounds
        return (ucoords >= 0) & (ucoords < image_width)

    def get_fov_bev(self, cam_record_data):
        #print("cam_record_data['cam_token']", cam_record_data['cam_token'])
        label_path = os.path.join(self.dataset_dir, "map-labels-v1.3", cam_record_data['cam_token'] + ".png")
        encoded_labels = to_tensor(Image.open(label_path)).long()

        NUSCENES_CLASS_NAMES = ['drivable_area', 'ped_crossing', 'walkway', 'carpark_area', 'bus',
                                'bicycle', 'car', 'construction_vehicle', 'motorcycle', 'trailer', 'truck', 'pedestrian', 'traffic_cone', 'barrier']

        # Decode to binary labels
        
        num_class = len(NUSCENES_CLASS_NAMES)
        labels = decode_binary_labels(encoded_labels, num_class + 1)
        # labels, mask = labels[:-1], ~labels[-1]
         

        # 训练时，使用v1.3，应用掩码，bev生成图片时，使用v1.3，应用掩码； bev分割时，使用v1.2不应用掩码
        # 对前14个通道应用掩码
        mask = ~labels[-1] 
        masked_labels = labels[:-1] * mask
        labels = masked_labels




        # 上下翻转操作！！！
        # bev = torch.flipud(labels.permute(1, 2, 0))
        bev = labels.permute(1, 2, 0)
        bev = bev.numpy()
        bev = (bev / 1.0).astype(np.float32)
        # channel_sums = np.sum(bev, axis=2)
        # print("Unique sums:", np.unique(channel_sums))
        # print(bev.shape)

        bev_resized = cv2.resize(bev, (self.cam_res[0], self.cam_res[1]), interpolation=cv2.INTER_LINEAR)
        # bev_resized = cv2.resize(bev, (256, 256), interpolation=cv2.INTER_LINEAR)

        # 宽度填充
        # new_width = 448
        # current_width = bev_resized.shape[1]  # 256
        # padding = (new_width - current_width) // 2  # 每侧填充量
        # left_pad = padding
        # right_pad = padding

        # # 检查是否需要不对称填充（如果总填充为奇数）
        # if (new_width - current_width) % 2 != 0:
        #     right_pad += 1  # 右侧多填1像素

        # # 应用填充：高度不填充，宽度两侧填充，通道不填充
        # bev_resized = np.pad(bev_resized, 
        #                     ((0, 0), (left_pad, right_pad), (0, 0)),  # 填充模式
        #                     mode='constant', 
        #                     constant_values=0)
    
        

        bev_resized = (bev_resized > 0.5)
        bev_resized = bev_resized.astype(np.float32)
        bev_resized = 2 * bev_resized - 1 
        # bev_resized = bev_resized[..., :6]
        # bev_resized = bev_resized[..., [1, 4, 5]]
        # bev_resized = normalized_data[..., [0, 2, 6]]

        return bev_resized

    def get_bev(self, cam_record, sample_token, sample_record, lidar_record, map_name):

        egolidar = self.nusc.get("ego_pose", lidar_record["ego_pose_token"])

        world_from_egolidarflat = parse_pose(egolidar, flat=True)
        egolidarflat_from_world = parse_pose(egolidar, flat=True, inv=True)

        cam_record_data = {}
        cam = self.nusc.get("calibrated_sensor", cam_record["calibrated_sensor_token"])
        # cam_record_data["token"] = sample_token
        cam_record_data["pose"] = world_from_egolidarflat.tolist()
        cam_record_data["pose_inverse"] = egolidarflat_from_world.tolist()
        cam_record_data["scene"] = self.nusc.get("scene", sample_record["scene_token"])["name"]
        cam_record_data["cam_token"] = cam_record['token']
        cam_record_data["intrinsic"] = cam["camera_intrinsic"]


        bev = None
        
        #print("running get_standard_bev")
        bev = self.get_fov_bev(cam_record_data)
       
       

        return bev

    def get_box_2d(self, box, view: np.ndarray = np.eye(3), normalize: bool = False):
        """
        计算边界框的2D投影
        
        参数:
            box: 边界框对象
            view: 视图变换矩阵
            normalize: 是否归一化
        """
        # 使用传入的边界框对象
        corners = view_points(box.corners(), view, normalize=normalize)[:2, :]
        points = []

        def get_rect(selected_corners):
            _points = []
            prev = selected_corners[-1]
            for corner in selected_corners:
                _points.append((int(prev[0]), int(prev[1])))
                _points.append((int(corner[0]), int(corner[1])))
                prev = corner
            return _points

        # Draw the sides
        for i in range(4):
            points.append((int(corners.T[i][0]), int(corners.T[i][1])))
            points.append((int(corners.T[i + 4][0]), int(corners.T[i + 4][1])))

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        points += get_rect(corners.T[:4])
        points += get_rect(corners.T[4:])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        points.append((int(center_bottom[0]), int(center_bottom[1])))
        points.append((int(center_bottom_forward[0]), int(center_bottom_forward[1])))

        points = np.array(points)

        return np.concatenate([points.min(0), points.max(0)], axis=0)


    def get_front_cam_annotations(self, frame_token):
        """
        获取前视摄像头(CAM_FRONT)中汽车(car)和卡车(truck)的数量
        
        参数:
            frame_token: 当前帧的token
        
        返回:
            (car_count, truck_count): 汽车数量, 卡车数量
        """
        # 指定前视摄像头
        CAMERA_NAME = 'CAM_FRONT'
        
        # 获取前视摄像头的样本数据token
        cam_data_token = self.nusc.get('sample', frame_token)['data'][CAMERA_NAME]
        
        # 获取前视摄像头的图像数据和边界框
        _, boxes, _ = self.nusc.get_sample_data(
            cam_data_token, 
            box_vis_level=BoxVisibility.ANY  # 使用字符串常量避免导入问题
        )
        
        # 初始化汽车和卡车计数
        car_count = 0
        truck_count = 0
        
        # 处理每个边界框，只统计汽车和卡车
        for box in boxes:
            # 获取类别名称，确保小写以统一处理
            parts = box.name.split('.')
            category = parts[1].lower() if len(parts) > 1 else parts[0].lower()
            
            if category == "car":
                car_count += 1
            elif category == "truck":
                truck_count += 1
        
        # 直接返回汽车和卡车的数量
        return car_count, truck_count


    def _get_box_3d(self, frame_token, box_type='LIDAR_TOP'):
        # get 3d gt bboxes w.r.t LIDAR_TOP
        sample_data_token = self.nusc.get('sample', frame_token)['data'][box_type]

        sd_rec = self.nusc.get('sample_data', sample_data_token)

        s_rec = self.nusc.get('sample', sd_rec['sample_token'])

        cs_rec = self.nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_rec = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])

        ann_recs = [self.nusc.get('sample_annotation', token) for token in s_rec['anns']]

        # mmdet3d setting
        N_CLASSES = 10
        CLASSES = [
            'car',
            'truck',
            'construction',
            'bus',
            'trailer',
            'barrier',
            'motorcycle',
            'bicycle',
            'pedestrian',
            'trafficcone',
        ]

        boxes, labels = [], []
        for ann_rec in ann_recs:

            ann_rec['sample_annotation_token'] = ann_rec['token']
            ann_rec['sample_data_token'] = sample_data_token

            # anns in world coord sys
            box = self.nusc.get_box(ann_rec['token'])

            # world to car
            box.translate(-np.array(pose_rec['translation']))
            box.rotate(Quaternion(pose_rec['rotation']).inverse)

            # car to cam
            box.translate(-np.array(cs_rec['translation']))
            box.rotate(Quaternion(cs_rec['rotation']).inverse)

            if set(CLASSES).intersection(box.name.split('.')) != set():
                labels.append(CLASSES.index(set(CLASSES).intersection(box.name.split('.')).pop()))
            else:
                continue

            boxes.append(np.concatenate(
                [box.center, box.wlh, np.array([- box.orientation.yaw_pitch_roll[0] - np.pi / 2])], axis=-1
            ))

        gt_anns_3d = {
            'gt_bboxes_3d': np.stack(boxes).astype(np.float32) if len(boxes) > 0 else np.zeros((1, 7)),
            'gt_labels_3d': np.array(labels) if len(boxes) > 0 else np.array([0, ]),
            'ego2sensor_rotation': Quaternion(cs_rec['rotation']).inverse,  #
            'ego2sensor_translation': - np.array(cs_rec['translation']),
        }
        return gt_anns_3d










    def get_all_camera_images(self, cam_record, sample_token, sample_record, lidar_record, map_name):
        """Returns images for all 6 cameras from keyframes"""

        egolidar = self.nusc.get('ego_pose', lidar_record['ego_pose_token'])

        world_from_egolidarflat = parse_pose(egolidar, flat=True)

        cam_idxs = []
        images = []
        angles = []
        intrinsics = []
        intrinsics_inv = []
        extrinsics_inv = []
        extrinsics = []
        cam_channels = []

        color_params = T.ColorJitter.get_params(brightness=self.color_transform.brightness, contrast=self.color_transform.contrast,
                                                saturation=self.color_transform.saturation, hue=self.color_transform.hue) if self.color_transform else None

        # bbx related
        max_num = 30
        bbxs = []
        masks = []
        rel_img_paths = []

        for cam_idx in self.select_cameras:
            mask_per_cam = np.zeros(max_num)

            cam_channel = Cameras.NUSCENES_CAMERAS[cam_idx]
            cam_channels.append(cam_channel)
            cam_token = sample_record['data'][cam_channel]

            cam_record = self.nusc.get('sample_data', cam_token)
            egocam = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
            cam = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])

            data_path, boxes, _ = self.nusc.get_sample_data(cam_token)

            cam_from_egocam = parse_pose(cam, inv=True)
            egocam_from_world = parse_pose(egocam, inv=True)

            E = cam_from_egocam @ egocam_from_world @ world_from_egolidarflat

            cam_idxs.append(cam_idx)

            extrinsics.append(E.tolist())
            extrinsics_inv.append(inv(E.tolist()))

            # I = cam['camera_intrinsic']
            # I = np.float32(I)
            # intrinsics_inv.append(inv(I))
            # intrinsics.append(I)

            image, center_angle_offset, center_pixel, camera_intrinsics = self.load_transform_image(cam_record["filename"], cam, color_params)

            bboxes_in_img = []
            # convert boundboxes coordinates to the image
            for i, bbx in enumerate(boxes):
                if not 'vehicle' in bbx.name:
                    continue
                corners = view_points(bbx.corners(),
                                      np.array(camera_intrinsics),
                                      normalize=True)[:2, :].T

                max_x = np.max(corners, axis=0)[0] / image.shape[1]
                max_y = np.max(corners, axis=0)[1] / image.shape[0]
                min_x = np.min(corners, axis=0)[0] / image.shape[1]
                min_y = np.min(corners, axis=0)[1] / image.shape[0]

                corners = np.clip(np.array([min_x, min_y, max_x, max_y], dtype=float), 0, 1)
                area = (corners[2] - corners[0]) * (corners[3] - corners[1])
                bboxes_in_img.append((corners, area))
                # debug purpose
                # cv2.imwrite('test.png', cv2.rectangle(image, (int(min_x*image.shape[1]), int(min_y*image.shape[0])), (int(max_x*image.shape[1]), int(max_y*image.shape[0])), (0, 0, 255), 2))

            if len(bboxes_in_img) > 0:
                bboxes_in_img = np.array(list(map(lambda x: x[0], nlargest(max_num, bboxes_in_img, key=lambda idx: idx[1]))))
            else:
                bboxes_in_img = np.full((max_num, 4), -1.0, dtype=np.float32)
            pad_len = max_num - bboxes_in_img.shape[0]
            bbx_per_cam = np.pad(bboxes_in_img, ((0, pad_len), (0, 0)), 'constant', constant_values=-1.0)

            mask_per_cam[:len(boxes)] = 1
            bbxs.append(bbx_per_cam)
            masks.append(mask_per_cam)

            I = camera_intrinsics
            I = np.float32(I)
            intrinsics_inv.append(inv(I))
            intrinsics.append(I)

            cam_angle = np.mod(quaternion_yaw(Quaternion(cam["rotation"]).rotation_matrix) + np.pi / 2, 2 * np.pi)
            angle = np.mod(cam_angle + center_angle_offset, 2 * np.pi).astype(np.float32)
            assert 0 <= angle <= 2 * np.pi
            images.append(image)
            angles.append(angle)

            rel_img_paths.append(os.path.relpath(Path(self.nusc.get_sample_data_path(cam_token)), self.dataset_dir))

        bbxs = np.stack(bbxs)
        masks = np.stack(masks)

        return {'image': np.stack(images),
                'angle': np.stack(angles).astype(np.float32),
                'intrinsics_inv': np.stack(intrinsics_inv).astype(np.float32),
                'extrinsics_inv': np.stack(extrinsics_inv).astype(np.float32),
                'intrinsics': np.stack(intrinsics).astype(np.float32),
                'extrinsics': np.stack(extrinsics).astype(np.float32),
                'cam_idx': np.stack(cam_idxs),
                'bbx': np.stack(bbxs),
                'bbx_mask': np.stack(masks),
                'image_paths': rel_img_paths,
                'sample_token': sample_token,
                'cam_name': cam_channels
                }

    def get_ego_pose(self, cam_record):
        sample_token = cam_record["sample_token"]
        sample_record = self.nusc.get("sample", sample_token)

        lidar_record = self.nusc.get("sample_data", sample_record["data"]["LIDAR_TOP"])
        egolidar = self.nusc.get("ego_pose", lidar_record["ego_pose_token"])
        # print("egolidar_rotation]", egolidar["rotation"])
        # print("egolidar_translation]", egolidar["translation"])

        world_from_egolidarflat = parse_pose(egolidar, flat=True)
        egolidarflat_from_world = parse_pose(egolidar, flat=True, inv=True)

        return world_from_egolidarflat, egolidarflat_from_world

    def get_single_image(self, cam_record, sample_token, sample_record, lidar_record, map_name):
        cam = self.nusc.get("calibrated_sensor", cam_record["calibrated_sensor_token"])
        image, center_angle_offset, center_pixel, camera_intrinsics = self.load_transform_image(cam_record["filename"], cam)
        # print("image.size", image.shape)

        # extrincs for this certain camera
        egocam = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
        cam_from_egocam = parse_pose(cam, inv=True)
        egocam_from_world = parse_pose(egocam, inv=True)
        world_from_egolidarflat, egolidarflat_from_world = self.get_ego_pose(cam_record)
        # print("world_from_egolidarflat",world_from_egolidarflat)
        # print("cam_from_egocam",cam_from_egocam)
        # print("egocam_from_world",egocam_from_world)
        # print("egocam_rotation]", egocam["rotation"])
        # print("egocam_translation]", egocam["translation"])
        
        # front camera用E = cam_from_egocam的效果和E = cam_from_egocam @ egocam_from_world @ world_from_egolidarflat一样
        # E = cam_from_egocam 
        E = cam_from_egocam @ egocam_from_world @ world_from_egolidarflat

        cam_angle = np.mod(quaternion_yaw(Quaternion(cam["rotation"]).rotation_matrix) + np.pi / 2, 2 * np.pi)

        car_count, truck_count = self.get_front_cam_annotations(sample_token)

        ret = {
            "channel": self.nusc.get("sensor", cam["sensor_token"])["channel"],
            "cam_angle": cam_angle,
            "cam": cam,
            "token": cam_record["calibrated_sensor_token"],
        }

        angle = np.mod(cam_angle + center_angle_offset, 2 * np.pi).astype(np.float32)
        assert 0 <= angle <= 2 * np.pi
        camera_intrinsic = np.array(camera_intrinsics)
        rotation = np.array(cam["rotation"])
        translation = np.array(cam["translation"])
        intrinsics_origin = np.array(cam["camera_intrinsic"])

        # print("translation is========= ", cam["translation"] )
        # print("intrinsics_origin is========= ", cam["camera_intrinsic"] )
        # print("rotation is========= ", cam["rotation"] )
        egolidar = self.nusc.get("ego_pose", lidar_record["ego_pose_token"])

        world_from_egolidarflat = parse_pose(egolidar, flat=True)
        egolidarflat_from_world = parse_pose(egolidar, flat=True, inv=True)

        cam_record_data = {}
        # cam_record_data["token"] = sample_token
        cam_record_data["pose"] = world_from_egolidarflat.tolist()
        cam_record_data["pose_inverse"] = egolidarflat_from_world.tolist()
        

        ret = {**ret, "image": image, "angle": angle,
               "intrinsics_inv": inv(camera_intrinsic),
               "extrinsics_inv": inv(E.tolist()),
               "rotation": rotation,
               "translation": translation,
               "intrinsics_origin": camera_intrinsic,
               "pose_inverse":cam_record_data["pose_inverse"],
               "E": E,
               "car_count": car_count,
               "truck_count": truck_count
                    
               
               }

        if self.single_camera_only:
            sample_token = cam_record["sample_token"]
            sample_record = self.nusc.get("sample", sample_token)
            image_paths = []

            for cam in self.CAMERAS:
                cam_channel = Cameras.NUSCENES_CAMERAS[cam]
                cam_token = sample_record["data"][cam_channel]
                full_path = Path(self.nusc.get_sample_data_path(cam_token))
                image_paths.append(str(full_path))
                intrinsics = self.nusc.get("calibrated_sensor", self.nusc.get("sample_data", cam_token)["calibrated_sensor_token"])["camera_intrinsic"]

            data = {**data, "image_paths": image_paths}

        return ret

    def get_cam(self, *args):
        if self.return_all_cams:
            ret = self.get_all_camera_images(*args)
        else:
            ret = self.get_single_image(*args)

        return ret

    def get_records(self, idx):
        if self.only_keyframes:
            sample_data_token = self.images_data[idx]
           # print("sample_data_token",sample_data_token)
            #print("idx",idx)
            cam_record = self.nusc.get("sample_data", sample_data_token)
            sample_token = cam_record["sample_token"]
            sample_record = self.nusc.get("sample", sample_token)
            log_token = self.nusc.get("scene", sample_record['scene_token'])['log_token']
            map_name = self.nusc.get("log", log_token)["location"]

            lidar_record = self.nusc.get("sample_data", sample_record["data"]["LIDAR_TOP"])
        else:
            lidar_record = self.nusc.get("sample_data", self.images[idx]["LIDAR_TOP"])
            nearest_sample_token = self.nusc.get("sample_data", lidar_record['token'])['sample_token']
            cam_record = lidar_record
            sample_token = str(lidar_record['timestamp']) + "_" + lidar_record['token']
            sample_record = {'data': self.images[idx], 'scene_token':  self.nusc.get("sample", self.nusc.get("sample_data", lidar_record['token'])['sample_token'])['scene_token']}
            map_name = None

        return cam_record, sample_token, sample_record, lidar_record, map_name

    def get_seg_data(self, data, record):
        layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
        if "return_render_img" in self.kwargs:
            sample_token = record[1]
            data_dir = Path("/data1/datasets/nuscenes_rendered")
            render_samples = []

            for cam_idx in self.select_cameras:
                camera_name = Cameras.NUSCENES_CAMERAS[cam_idx]
                file_name = data_dir / f"{sample_token}_{camera_name}.npz"
                if exists(file_name):
                    render_sample_ = np.load(file_name)
                    render_sample = render_sample_.f.arr_0
                else:
                    image_sample = np.zeros((900, 900, 3), dtype=np.uint8)
                    # check the correct map api
                    nusc_map = None
                    for scene_name, api in self.nusc_map.items():
                        if api.map_name == record[-1]:
                            nusc_map = api
                    render_sample, ax = render_helper.render_map_in_image(self.nusc,
                                                                          nusc_map,
                                                                          sample_token,
                                                                          image_sample,
                                                                          layer_names=layer_names,
                                                                          camera_channel=camera_name)
                    render_sample = cv2.resize(render_sample, (256, 256))
                    np.savez_compressed(file_name, render_sample)

                render_samples.append(render_sample)
            render_samples = np.array(render_samples)
            data = {**data, "render_samples": render_samples}

        if "return_seg_img" in self.kwargs:
 # preds = []
            # imgs = []
            # for path in data['image_paths']:
            #     file_path = (Path('/data1/datasets/nuscenes_cityscapes_v3') / path).with_suffix('.npz')
            #     pred = np.load(file_path)['pred']
            #     preds.append(pred)
            # 
            #     img = Image.open(self.dataset_dir / path)
            #     img = img.crop((350, 0, 1250, 900))
            #     img = img.resize((256, 256))
            #     img = np.asarray(img)
            #     img = self.cam_transform(torch.from_numpy(img / 255).permute(2, 0, 1)).permute(1, 2, 0)
            #     imgs.append(img)
            # 
            # seg_mask = np.stack(preds).transpose(1, 2, 0)
            # data['image_gt'] = data['image'].copy()

            CAM_DATA = {'CAM_FRONT': (1266.417203046554, 1266.417203046554, 0.005684811144346602),
                        'CAM_BACK': (809.2209905677063, 809.2209905677063, 3.1391709219861887),
                        'CAM_FRONT_RIGHT': (1260.8474446004698, 1260.8474446004698, 5.298742851167251),
                        'CAM_FRONT_LEFT': (1272.5979470598488, 1272.5979470598488, 0.9627404474321728),
                        'CAM_BACK_RIGHT': (1259.5137405846733, 1259.5137405846733, 4.349372983905386),
                        'CAM_BACK_LEFT': (1256.7414812095406, 1256.7414812095406, 1.895431863668132)}
            warped_imgs = []
            for cam_idx in self.select_cameras:
                cam_channel = Cameras.NUSCENES_CAMERAS[cam_idx]

                rotated_im = util.rot_img(rearrange(data['segmentation'], 'h w c -> () c h w'),
                                          -CAM_DATA[cam_channel][2], dtype=torch.FloatTensor)
                rotated_im = rearrange(rotated_im, '() c h w -> h w c')
                rotated_im = torch.from_numpy(polar_transform(rotated_im.numpy()))
                #warped_imgs.append(rotated_im)
                data['warped_imgs'] = rotated_im

                # viz_bev(rearrange(rotated_im, 'h w c -> c h w')).save(cam_channel)
            # seg_mask = torch.from_numpy(seg_mask)
            # data['segmentation'] = torch.from_numpy(polar_transform(data['segmentation'].numpy()))
            # data = {**data, "camera_segmentation": seg_mask}

        return data


    def __getitem__(self, idx):
        record = self.get_records(idx)
        data = {"dataset": "nuscenes"}

        if self.return_bev_img:
            bev = torch.Tensor(self.get_bev(*record))
           # print("bev.shape=========", bev.shape)
            data = {**data, "segmentation": bev}

        if self.return_cam_img:
            data = {**data, **self.get_cam(*record)}

        data = self.get_seg_data(data, record)


        return data

    def __len__(self):
        return len(self.images_data)

    def viz_item(self, batch: dict, idx):
        imgs = raw_output_data_bev_grid(batch)
        for i in range(len(imgs)):
            os.makedirs(f'output', exist_ok=True)
            tiled_img = imgs[i]
            tiled_img = tiled_img.resize((tiled_img.size[0]//2,tiled_img.size[1]//2), Image.Resampling.LANCZOS)
            tiled_img.save(f'output/{batch["sample_token"][i]}.png', quality=95)

    def get_cam_angles(self, batch: dict):
        for angle, cam_name, intrinsics in zip(batch['angle'][0], batch['cam_name'], batch['intrinsics'][0]):
            print(cam_name[0], np.rad2deg(angle))

            print(np.rad2deg(2 * np.arctan(1600 / (2 * intrinsics[0, 0]))))

    def save_cam_data(self, data: dict):
        torch.save({k: v for k, v in data.items() if "intrinsics" in k or "extrinsics" in k}, 'nuscenes_cam_data.pt')


if __name__ == "__main__":
    CITYSCAPES_CONFIG = {
        'split': 0,
        'return_cam_img': False,
        'return_bev_img': False,
        'return_all_cams': True,
        'stage_2_training': True,
        'metadrive_compatible_v2': True,
        'non_square_images': True,
        'mini_dataset': True,
        'only_keyframes': True,
        'cam_res': (256, 256),
        'augment_cam_img': False,
        'augment_bev_img': False,
        # "square_images_cityscapes": True,
        # 'generate_split': (1, 0, 1),
        "return_seg_img": False,
        "return_render_img": True
        # "normalize_cam_img": False
    }

    RENDER_CONFIG = {
        'split': 0,
        'return_cam_img': True,
        'return_bev_img': True,
        'return_all_cams': True,
        'stage_2_training': True,
        'metadrive_compatible_v2': True,
        'non_square_images': True,
        'mini_dataset': True,
        'only_keyframes': True,
        'cam_res': (224, 400),
        'augment_cam_img': False,
        'augment_bev_img': False,
        # "square_images_cityscapes": True,
        # 'generate_split': (1, 0, 1),
        # "return_seg_img": False,
        "return_seg_img": True
        # "normalize_cam_img": False
    }

    TEST_CONFIG = {
        'split': 0,
        'return_cam_img': True,
        'return_bev_img': True,
        'return_all_cams': True,
        'stage_2_training': True,
        'metadrive_compatible_v2': True,
        'non_square_images': True,
        'mini_dataset': True,
        'only_keyframes': True,
        'cam_res': (224, 400),
        'augment_cam_img': False,
        'augment_bev_img': False,
        # "square_images_cityscapes": True,
        # 'generate_split': (1, 0, 1),
        # "return_seg_img": False,
        "return_seg_img": True
        # "normalize_cam_img": False
    }

    STAGE_2_CONFIG = {
        'return_cam_img': True,
        'return_bev_img': True,
        'return_all_cams': True,
        'stage_2_training': True,
        'metadrive_compatible_v2': True,
        'non_square_images': True,
        'mini_dataset': True,
        'only_keyframes': False,
        'cam_res': (224, 400),
    }
    TRAIN_CONFIG = {
        **STAGE_2_CONFIG,
        'split': 0,
        'augment_cam_img': True,
        'augment_bev_img': True,
    }

    VAL_CONFIG = {
        **STAGE_2_CONFIG,
        'split': 1,
        'only_keyframes': False,
    }

    VIZ_CONFIG = {
        **STAGE_2_CONFIG,
        'split': 0,
        'cam_res': (900, 1600),
        'mini_dataset': False,
    }

    GEN_CONFIG = {
        'split': 0,
        # 'frozen_image_tokens': 'pretrained/test_image_tokens.txt',
        # 'frozen_scene_tokens': 'data/nonoverlapping_scene_tokens.txt',
        # 'only_keyframes': True,
        'return_cam_img': True,
        # 'generate_split': (1, 0, 10),
        'return_bev_img': True,
        'augment_cam_img': True,
        'augment_bev_img': True,
        'mini_dataset': True,
        'metadrive_compatible': False,
        'cam_res': False,
        'dataset_dir': "/data01/hwj/autodrive/affordance/dataset/nuscenes",

    }

    ABLATION_CONFIG = {
        'split': 1,
        'return_cam_img': True,
        'return_bev_img': True,
        'return_all_cams': True,
        'metadrive_compatible_v2': True,
        'stage_2_training': True,
        'non_square_images': True,
        'only_keyframes': True,
        'frozen_image_tokens': 'pretrained/ablation_samples.pkl',
        'eval_generate': SAVE_DATA_DIR / 'ablations/causal_only',
        'mini_dataset': False,
        'cam_res': (224, 400),
        'generate_split': (1, 0, 10),
        'select_cameras': (0, 2, 3)
    }

    METADRIVE_INFERENCE_CONFIG = {
        'split': 0,
        'return_cam_img': False,
        'return_bev_img': True,
        'augment_bev_img': False,
        'mini_dataset': True,
        'metadrive_compatible': True,
        'metadrive_compatible_v2': True,
        'only_keyframes': True,
    }

    output_dir = './log_dataset'

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        dataset = NuScenesDataset(**GEN_CONFIG)
        batch_iter = DataLoader(dataset, batch_size=8, num_workers=0, shuffle=True, pin_memory=False)
        for i, data in tqdm(enumerate(batch_iter)):
            groundtruth = data['image'].cpu().numpy()
            for j in range(groundtruth.shape[0]):  
                # img = groundtruth[j].transpose(1, 2, 0)  
                img = groundtruth[j]
                img = Image.fromarray(img.astype('uint8'))  

                
                file_name = f"image_{i * 8 + j}.jpg"  
                img.save(os.path.join(output_dir, file_name))

            gt_mask = data['segmentation'].cpu().numpy()
            for j in range(gt_mask.shape[0]):
                file_name = f"bev_{i * 8 + j}.jpg"  
                viz_bev(gt_mask[j]).pil.save(
                    os.path.join(output_dir,
                                 'w_cfg',
                                 file_name,
                                 ))
            breakpoint()
            # Im(data['render_samples']).save('render')
            # Im(data['image'][0]).denormalize().save('image.png')
            # breakpoint()
            # dataset.viz_item(data, i)
    except Exception as e:
        print(e)
        breakpoint()
