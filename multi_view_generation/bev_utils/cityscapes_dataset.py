import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from typing import Dict, Union, Tuple, Optional
from multi_view_generation.bev_utils import util
import cv2
from torch.utils.data import DataLoader
from multi_view_generation.bev_utils import bev_pixels_to_cam , render_map_in_image
from multi_view_generation.bev_utils import viz_bev
from image_utils import Im
import matplotlib.pyplot as plt


class CityscapesDataset(Dataset):
    def __init__(self, transform: bool = True, cam_res: Tuple[int, int] = (256, 256)):
        """
        初始化Cityscapes数据集类

        Args:
            image_dir (str): 图像文件路径（例如 ".../leftImg8bit/train"）
            gt_dir (str): 标注文件路径（例如 ".../gtFine/train"）
            transform (callable, optional): 同时处理图像和标签的转换函数
        """
        self.mutichannel_mode = 0
        self.transform = transform
        image_dir = '/opt/data/private/hwj_autodrive/autodrive/dataset/leftImg8bit/train'
        gt_dir = '/opt/data/private/hwj_autodrive/autodrive/dataset/gtFine/train'
        self.files = self._load_file_paths(image_dir, gt_dir)
        self.cam_res = cam_res

        # 验证至少存在一个有效文件
        if len(self.files) == 0:
            raise RuntimeError(f"No valid files found in {image_dir}")
        if not os.path.isfile(self.files[0][1]):
            raise FileNotFoundError(
                "Label file not found. Please generate labelTrainIds.png using "
                "cityscapes scripts. First file path: {}".format(self.files[0][1])
            )

    def _load_file_paths(self, image_dir, gt_dir):
        """
        加载所有有效的图像-标签文件对

        Args:
            image_dir (str): 图像根目录
            gt_dir (str): 标注根目录

        Returns:
            list: 包含(image_path, label_path)元组的列表
        """
        file_pairs = []

        # 遍历城市文件夹
        for city in os.listdir(image_dir):
            city_img_dir = os.path.join(image_dir, city)
            city_gt_dir = os.path.join(gt_dir, city)

            # 遍历城市目录中的图像文件
            for img_name in os.listdir(city_img_dir):
                if img_name.endswith("_leftImg8bit.png"):
                    # 构建对应的标注文件名
                    base_name = img_name.split("_leftImg8bit.png")[0]
                    label_name = base_name + "_gtFine_labelTrainIds.png"

                    img_path = os.path.join(city_img_dir, img_name)
                    label_path = os.path.join(city_gt_dir, label_name)

                    # 仅添加存在的文件对
                    if os.path.exists(label_path):
                        file_pairs.append((img_path, label_path))

        return file_pairs

    def load_transform_image(self, img):

        img_resize = img.resize((self.cam_res[0], self.cam_res[1]), Image.Resampling.BICUBIC)
        im_to_process = np.asarray(img)
        img_resize = np.asarray(img_resize)
        center_pixel = im_to_process.shape[1] // 2
        self.max_h_pre_crop = self.cam_res[0]

        if self.max_h_pre_crop is not None:
            scale = self.max_h_pre_crop / float(max(im_to_process.shape[1],
                                                    im_to_process.shape[0]))

            im_to_process = util.smallest_max_size(im_to_process,
                                                   self.max_h_pre_crop)

        cam_transform = A.Compose([])
        cam_transform = A.Compose(
            [
                cam_transform,
                # A.Normalize(mean=(0.4265, 0.4489, 0.4769), std=(0.2053, 0.2206, 0.2578), max_pixel_value=255.0),
                A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0),
            ]
        )
        cam_transform = A.Compose([cam_transform, A.CenterCrop(height=self.cam_res[0], width=self.cam_res[1])])

        if cam_transform is not None:
            transformed_cam_img = cam_transform(image=img_resize)["image"]
        else:
            transformed_cam_img = im_to_process

        # reform the intrinsics
        img_w, img_h = img.size[1], img.size[0]

        return transformed_cam_img
    
    def load_transform_label_12c(self, label):
        label_to_process = np.asarray(label)

        # 创建全零数组用于存储新标签
        new_array = np.zeros_like(label_to_process)

        # 根据原标签值重新映射 (扩展为12个类别)
        new_array[label_to_process == 0] = 1  # 原0转1
        new_array[label_to_process == 1] = 2  # 原1转2
        new_array[label_to_process == 2] = 3  # 原2转3
        new_array[label_to_process == 3] = 4  # 原3转4
        new_array[label_to_process == 4] = 5  # 原4转5
        new_array[label_to_process == 5] = 6  # 原5转6
        new_array[label_to_process == 6] = 7  # 原6转7
        new_array[label_to_process == 7] = 8  # 原7转8
        new_array[label_to_process == 11] = 9  # 原11转9
        new_array[label_to_process == 12] = 10  # 原12转10
        new_array[label_to_process == 10] = 11  # 原10转11
        new_array[label_to_process == 13] = 12  # 原13转12

        if self.mutichannel_mode == 0:
            # 类别映射到通道的映射关系 (12个类别对应12个通道)
            class_mapping = {
                1: 0,  # new_array=1 -> 通道0
                2: 1,  # new_array=2 -> 通道1
                3: 2,  # new_array=3 -> 通道2
                4: 3,  # new_array=4 -> 通道3
                5: 4,  # new_array=5 -> 通道4
                6: 5,  # new_array=6 -> 通道5
                7: 6,  # new_array=7 -> 通道6
                8: 7,  # new_array=8 -> 通道7
                9: 8,  # new_array=9 -> 通道8
                10: 9,  # new_array=10 -> 通道9
                11: 10,  # new_array=11 -> 通道10
                12: 11  # new_array=12 -> 通道11
            }

            # 创建多通道二值图（12通道）
            binary_mask = np.zeros((label_to_process.shape[0],
                                    label_to_process.shape[1],
                                    12), dtype=np.uint8)  # 改为12通道

            # 遍历每个类别，填充对应的通道
            for value, channel in class_mapping.items():
                binary_mask[new_array == value, channel] = 1

            new_array = binary_mask

        # 归一化并转换到[-1, 1]范围
        new_array = (new_array / 1.0).astype(np.float32)
        new_array = 2 * new_array - 1

        # 多通道图像的分通道resize处理
        if new_array.ndim == 3:  # 多通道情况
            resized_channels = []
            for c in range(new_array.shape[2]):
                channel_img = new_array[:, :, c]
                resized_channel = cv2.resize(channel_img,
                                             (self.cam_res[0], self.cam_res[1]),
                                             interpolation=cv2.INTER_LINEAR)
                resized_channels.append(resized_channel)
            label_resized = np.stack(resized_channels, axis=2)
        else:  # 单通道情况
            label_resized = cv2.resize(new_array,
                                       (self.cam_res[0], self.cam_res[1]),
                                       interpolation=cv2.INTER_LINEAR)

        return label_resized

    def load_transform_label(self, label):

        label_to_process = np.asarray(label)

        # 创建全零数组用于存储新标签
        # new_array = label_to_process
        new_array = np.zeros_like(label_to_process)

        # 根据原标签值重新映射
        new_array[label_to_process == 0] = 1  # 原0转1
        new_array[label_to_process == 2] = 2  # 原2保持2
        new_array[label_to_process == 13] = 3  # 原13转3

        if self.mutichannel_mode == 0:
            class_mapping = {
                1: 0,  # new_array=1 -> 通道0（类别1）
                2: 1,  # new_array=2 -> 通道1（类别2）
                3: 2  # new_array=3 -> 通道2（类别3）
            }

            # 创建多通道二值图（3通道）
            binary_mask = np.zeros((label_to_process.shape[0], label_to_process.shape[1], 3), dtype=np.uint8)

            # 遍历每个类别，填充对应的通道
            for value, channel in class_mapping.items():
                binary_mask[new_array == value, channel] = 1
            new_array = binary_mask

        new_array = (new_array / 1.0).astype(np.float32)
        new_array = 2 * new_array - 1 
        # hwj modified
        label_resized = cv2.resize(new_array, (self.cam_res[0], self.cam_res[1]), interpolation=cv2.INTER_LINEAR)

        return label_resized


    def vis_singlec(self, new_array, save_path):

        vis_array = new_array.astype(np.uint8)

        label_img = Image.fromarray(vis_array, mode='L')

        # 创建调色板（256种颜色，每个颜色3字节RGB值）
        palette = [0] * 768  # 初始化全黑调色板

        # 定义颜色映射（索引 -> RGB）
        palette[0 * 3: 1 * 3] = [0, 0, 0]  # 0: 黑色（背景）
        palette[1 * 3: 2 * 3] = [255, 0, 0]  # 1: 红色
        palette[2 * 3: 3 * 3] = [0, 255, 0]  # 2: 绿色
        palette[3 * 3: 4 * 3] = [0, 0, 255]  # 3: 蓝色

        # 转换为调色板模式并应用颜色
        label_p = label_img.convert('P')
        label_p.putpalette(palette)

        # 保存可视化结果
        label_p.convert('RGB').save(save_path)

    def visualize_label_image(self, new_array, output_path=None):
    # 读取标签图并转换为灰度数组
        label_array = new_array.astype(np.uint8)
    # 提取唯一标签值
        unique_labels = np.unique(label_array)
        num_labels = len(unique_labels)
    
        if num_labels == 0:
           raise ValueError("标签图中没有有效的标签值。")
    
    # 创建颜色映射（可替换为其他如'viridis'）
        cmap = plt.get_cmap('tab20', num_labels)
        colors = (cmap(np.arange(num_labels))[:, :3] * 255).astype(np.uint8)
    
    # 映射标签到颜色
        sorted_labels = np.sort(unique_labels)
        indices = np.searchsorted(sorted_labels, label_array)
        rgb_array = colors[indices]
    
    # 转换为PIL图像
        rgb_image = Image.fromarray(rgb_array)
    
    # 保存图片（如果指定了输出路径）
        if output_path is not None:
           rgb_image.save(output_path, format='PNG', compress_level=0)  # 无压缩保存
        
        return rgb_image


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        返回处理后的图像和标签张量

        Args:
            idx (int): 数据索引

        Returns:
            tuple: (image, label) 已应用转换的张量
        """
        img_path, label_path = self.files[idx]

        # 加载图像和标签
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')  # 单通道模式

        # 应用转换（例如数据增强）
        if self.transform:
            image = self.load_transform_image(image)
            label = self.load_transform_label_12c(label)

        bev = torch.Tensor(label)
        data = {"image": image}
        data = {**data, "segmentation": bev}

        return data


if __name__ == "__main__":

    output_dir_bev = './cityscapes_dataset_test/bev'
    output_dir_img = './cityscapes_dataset_test/img'
    label_path = '/opt/data/private/hwj_autodrive/autodrive/dataset/gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png'
    # with Image.open(label_path) as img:
    #     label = img.convert('L')  # 强制转为8位灰度（范围0-255）
    #     label_array = np.array(label)

    # # 获取唯一值并排序
    # unique_values = np.unique(label_array)

    # # 打印结果
    # print(f"Unique label values ({len(unique_values)} classes):")
    # print(np.sort(unique_values))

    dataset = CityscapesDataset(
    )

    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 迭代数据
    for i, batch in enumerate(dataloader):
        groundtruth = batch['image'].cpu().numpy()
        gt_mask = batch['segmentation'].cpu().numpy()
        for j in range(gt_mask.shape[0]):
            # print("type(groundtruth) is ", type(groundtruth[j]))
            print("groundtruth[j].shape is ", groundtruth[j].shape)
            print("gt_mask[j].shape is ", gt_mask[j].shape)
            file_name_img = f"img_{i * gt_mask.shape[0] + j}.jpg"
            file_name_bev = f"bev_{i * gt_mask.shape[0] + j}.jpg"
            img_c = ((groundtruth + 1) / 2).clip(0, 1)
            imgc = (img_c[j] * 255).astype(np.uint8)
            img = Im(imgc)
            img.save(os.path.join(output_dir_img, file_name_img))            
            
            if dataset.mutichannel_mode == 0:
                print(gt_mask.shape)
                viz_bev(gt_mask[j]).pil.save(
                             os.path.join(output_dir_bev,
                                         file_name_bev,
                                         ))
            
            else:
                dataset.vis_singlec(gt_mask[j],os.path.join(output_dir_bev,
                                         file_name_bev,
                                         ))

