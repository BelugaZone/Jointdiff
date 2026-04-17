import os
import sys
from pathlib import Path
_JOINTDIFF_ROOT = Path(__file__).resolve().parents[2]
if str(_JOINTDIFF_ROOT) not in sys.path:
    sys.path.insert(0, str(_JOINTDIFF_ROOT))
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
from pipeline.pipeline_jointdiff import StableDiffusionJointdiffPipeline
from utils.utils import blip2_cap, get_combined_filename, save_png_with_comment, set_jointdiff_config_inference, parse_schedule, process_images
from utils.load_utils import load_model_configs, load_jointdiff, load_blip_processor, load_annotator, annotator_dict, load_jointdiff_weights, load_scheduler
from multi_view_generation.bev_utils import viz_bev
from multi_view_generation.bev_utils import bev_pixels_to_cam , render_map_in_image, cam_pixels_to_bev
from utils.utils import get_prompt
from skimage.transform import resize

from accelerate import DistributedDataParallelKwargs
from utils.confusion import BinaryConfusionMatrix

confusion = {
    "0.5": BinaryConfusionMatrix(6),
}
threshold = 0.0
pred_valid = []
mask_seq = []

def encode_binary_labels(masks):
    bits = np.power(2, np.arange(len(masks), dtype=np.int32))
    return (masks.astype(np.int32) * bits.reshape(-1, 1, 1)).sum(0)

def process_segmentation(y_img_pil):
            E01 = np.array([
               [ 5.6847786e-03, -9.9998349e-01,  8.0507132e-04,  5.0603189e-03],
               [-5.6366678e-03, -8.3711528e-04, -9.9998379e-01,  1.5205332e+00],
               [ 9.9996793e-01,  5.6801485e-03, -5.6413338e-03, -1.6923035e+00],
               [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
                ])
            
            intrinsics01 = np.array([
                       [202.62675249, 0.0, 130.60272316],
                       [0.0, 360.22533776, 139.80645427],
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

def process_rgb(x_img_pil):
            if x_img_pil.shape[2] == 3:
                 x_tensor = ((x_img_pil + 1) / 2).clip(0, 1).cpu().numpy()
            else:
                 x_tensor = ((x_img_pil + 1) / 2).clip(0, 1).permute(1,2,0).cpu().numpy()
            x_tensor = (x_tensor * 255).astype(np.uint8)
            print('x_tensor',x_tensor.shape)
    
             # 创建PIL图像
            pil_img = Image.fromarray(x_tensor, 'RGB' if x_tensor.shape[-1] == 3 else 'L')
            # pil_img = pil_img.resize((448,256), Image.Resampling.BICUBIC)
            
            return pil_img

def get_prompt_old(masks):
        """
        检查批量的多通道二值掩码中各通道是否全零，返回两批格式化的描述字符串
        
        参数:
        masks: 输入的批量多通道二值掩码，形状为：
            [批次数, 通道数, 高度, 宽度] 或 
            [批次数, 高度, 宽度, 通道数] 或 
            [通道数, 高度, 宽度] (单样本)
            通道顺序：drivable_area, ped_crossing, walkway, carpark, car, truck
        
        返回:
        (x_strings, y_strings) 元组，每个元素为字符串列表，格式分别为：
            x_string: "a realistic driving scene with drivable_area，ped_crossing"
            y_string: "a seg mask with drivable_area，ped_crossing"
        """
        # 定义通道标签
        channel_labels = [
            "drivable_area",
            "ped_crossing",
            "walkway",
            "carpark",
            "car",
            "truck"
        ]
        
        # 确保输入是PyTorch张量
        if not isinstance(masks, torch.Tensor):
            masks = torch.tensor(masks, dtype=torch.float32)
        
        # 处理不同输入形状
        if masks.dim() == 3:  # 单样本输入 (C, H, W) 或 (H, W, C)
            masks = masks.unsqueeze(0)  # 添加批次维度
        
        # 转换值范围 (假设输入是-1到1)
        masks = ((masks + 1) / 2)
        
        # 调整通道维度到倒数第三位 (批次, 通道, H, W)
        if masks.shape[-1] == 6:  # 通道在最后一维 (B, H, W, C)
            masks = masks.permute(0, 3, 1, 2)  # 转换为 (B, C, H, W)
        elif masks.shape[1] != 6:  # 不是 (B, C, H, W) 格式
            raise ValueError(f"无效的输入形状: {masks.shape}")
        
        # 识别批处理中的非全零通道
        batch_size = masks.shape[0]
        x_strings = []
        y_strings = []
        
        for i in range(batch_size):
            sample_mask = masks[i]
            nonzero_labels = []
            
            for c in range(6):
                channel = sample_mask[c]
                if not torch.all(channel == 0):
                    nonzero_labels.append(channel_labels[c])
            
            # 为当前样本构建描述字符串
            if nonzero_labels:
                labels_str = "，".join(nonzero_labels)
                x_strings.append(f"a realistic driving scene with {labels_str}")
                y_strings.append(f"a seg mask with {labels_str}")
            else:
                x_strings.append("a realistic driving scene")
                y_strings.append("a seg mask")
        
        return x_strings, y_strings



test_dataset_nuscenes = NuScenesDataset(
        split=1,  # 必需的位置参数
        return_cam_img=True,
        return_bev_img=True,
        augment_cam_img=False,
        augment_bev_img=False,
        return_seg_img=True,
        return_all_cams=False,
        mini_dataset=False,
        metadrive_compatible=False,
        cam_res=(256,256),
        dataset_dir="/opt/data/private/nuscenes"
    )

dataset_length = len(test_dataset_nuscenes)

test_dataloader = torch.utils.data.DataLoader(
        test_dataset_nuscenes ,
        shuffle=False,
        batch_size=8,
        num_workers=1,
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():

    model_configs = load_model_configs()
    cur_model = "depth"
    pipeline_class = StableDiffusionJointdiffPipeline
    pipe = load_jointdiff(pipeline_class, model_configs[cur_model])    
    total_samples = len(test_dataloader.dataset)

    log_file = open("/opt/data/private/hwj_autodrive/jointdiff/eval_metric_visual/512_dat_result_metric/experiment_600sample_updown_cross_single512_15000.txt", "w")

    MAX_SAMPLES = 600
    
    # 创建主进度条（样本级别）
    main_pbar = tqdm(
        total=total_samples, 
        desc="Processing samples", 
        position=0,
        file=log_file  # 将进度条输出重定向到文件
    )

    processed_samples = 0

    for step, batch in enumerate(test_dataloader):
        if processed_samples >= MAX_SAMPLES:
                break

        for i in range(batch["image"].shape[0]): 
            if processed_samples >= MAX_SAMPLES:
                break 
            height = 256
            width = 256
            # image_input = Image.open(image_input).convert("RGB")

            #####涉及修改pipline 1346行图像预处理
            c = batch["segmentation"][:,:,:,:6]


            truck_count = batch['truck_count'][i]
            car_count = batch['car_count'][i]
            camera_intrinsic = np.array([
                       [202.62675249, 0.0, 130.60272316],
                       [0.0, 360.22533776, 139.80645427],
                       [0.0, 0.0, 1.0]
                       ])

            camera_intrinsic_tensor = torch.tensor(camera_intrinsic, dtype=c.dtype)

            # 添加批次维度并复制
            camera_intrinsic_batch = camera_intrinsic_tensor.unsqueeze(0).repeat(2, 1, 1)
            camera_intrinsic_batch = camera_intrinsic_batch.to(device)
            # print('truck_count',truck_count)
            # print('car_count',car_count)

            # print('prompts',prompts_new)


            # 对应文本获取

            seg = batch["segmentation"][i][:,:,:6]

            x_caption , y_caption = get_prompt_old(seg)

            # x_caption , y_caption = get_prompt(seg,car_count.unsqueeze(0),car_count.unsqueeze(0))
        
            
            x_prompt = x_caption[0]
            y_prompt = y_caption[0]
            
            
            negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

           

            prompts = [x_prompt,y_prompt]
            print('prompts',prompts)


            
            cond_input = c[i].permute(2,0,1)
            image = batch["image"][i].permute(2,0,1)
            cond_input = torch.zeros_like(cond_input)

            image_input = torch.zeros_like(cond_input)

            # cond_input = Image.new('RGB', (width, height), (255, 255, 255))
        
            # cond_input = Image.open(cond_input)
            image_mask = Image.new('RGB', (width, height), (255, 255, 255))
            cond_mask = Image.new('RGB', (width, height), (255, 255, 255))

            init_images = [image_input,  cond_input]  #C, H, W]

            masks = [image_mask] + [cond_mask] 
            
            masks = masks

            image_noise_strength = 1.0
            cond_noise_strength = 1.0

            sample_schedule = [[(1 - image_noise_strength, 1), (1 - cond_noise_strength, 1), 0]]

            batch_size = 1

            num_input = batch_size * 2

            joint_scale = 1.0

            num_inference_step = 25

            seed = random.randint(0, 2**32 - 1)

            guidance_scale = 4.5

            cond_guidance_scale = 2

            inputs_config = {
                "num_input": num_input,
                "schedules": sample_schedule,
                "pairs": [(j, j+batch_size, joint_scale, joint_scale, cur_model) for j in range(batch_size)]
            }
            debug = False
            set_jointdiff_config_inference(pipe.unet, inputs_config["pairs"], batch_size = inputs_config["num_input"], debug=debug)
            load_scheduler(pipe, mode = "ead")

            generator = torch.Generator(device="cuda")
            generator = generator.manual_seed(seed)

            trigger_word = "depth"
            # prompt = 'none'
            # prompts = [prompt] * num_input
            # negative_prompt  = 'none'
            if negative_prompt is not None:
                negative_prompts = [negative_prompt] * num_input
            
            sample_schedule = parse_schedule(inputs_config["schedules"], num_inference_step)
            enable_joint_attn = True
            patch.set_joint_attention(pipe.unet, enable = enable_joint_attn)



            pipe.to("cuda")

            inference_config = {
                "prompt" : prompts,
                "image" : init_images,
                "mask_image" : masks,
                "height" : height,
                "width" : width,
                "num_inference_steps" : num_inference_step,
                "guidance_scale" : guidance_scale,
                "strength" : 1.0,
                "negative_prompt" : negative_prompts,
                "eta" : 1.0,
                "sample_schedule" : sample_schedule,
                "cond_guidance_scale" : cond_guidance_scale,
                "xlen" : batch_size,
                "generator":generator,
                "camera_intrinsic":camera_intrinsic_batch,
            }


            with torch.autocast(device_type = "cuda", dtype = torch.float16):
                images = pipe(**inference_config).images

            split_tensors = torch.split(images, split_size_or_sections=1, dim=0)
            
            x_images = split_tensors[0]
            x_images = x_images[:,:3,:,:]
            y_images = split_tensors[1]
            print('x_images',x_images.shape)
           

            bev_cond , bev_project, labels = process_segmentation(y_images[0])
            img_gen = process_rgb(x_images[0]) 
            img_gt = process_rgb(image)

            # y_tensor = ((y_images + 1) / 2).clip(0, 1)
            
            # gt_all = batch["segmentation"][i].permute(2,0,1).to(device).detach()
            # gt_all = ((gt_all + 1) / 2).clip(0, 1)
            # gt_mask = gt_all[-1:, :, :]
            # gt = gt_all[:6, :, :]


            # # pred = (y_tensor > 0.5).float().squeeze().cpu().numpy()
            # pred = y_tensor.float().squeeze().cpu().numpy()
            # print("pred.shape",pred.shape)
            # gt = (gt > 0).float().squeeze().cpu().numpy()

            # gt_mask = 1 - gt_mask
            # gt_mask = (gt_mask > 0.5).float().squeeze().cpu().numpy()
            # mask_expanded = gt_mask[np.newaxis, :, :]  # 形状变为 (1, H, W)
            # # 将掩码复制到每个通道
            # mask_expanded = np.repeat(mask_expanded, pred.shape[0], axis=0)
            # masked_pred = np.where(mask_expanded > 0, pred, 0)
            # masked_pred = viz_bev(masked_pred).pil

            # /opt/data/private/hwj_autodrive/jointdiff/AAstanderd_imggen_joint/no_viewmask/imggen_dat_bce_85000 scale=4
           
            # /opt/data/private/hwj_autodrive/jointdiff/AAstanderd_imggen_joint/pon_split/imggen_pure_25000_timeplan
            #/opt/data/private/hwj_autodrive/jointdiff/AAstanderd_imggen/ponsplit/25000rebev2img/imggen_full_doubleloss0.01_25000
          

            output_dir_imgen= './AAstanderd_imggen_joint/pon_split/imggen_full_25000_0.01loss_scale=2/img_gen'
            output_dir_bev_cond = './AAstanderd_imggen_joint/pon_split/imggen_full_25000_0.01loss_scale=1/bev_cond'
            output_dir_gen_project = './AAstanderd_imggen_joint/pon_split/imggen_full_25000_0.01loss_scale=1/gen_project'
            output_dir_imggt = './AAstanderd_imggen_joint/pon_split/imggen_full_25000_0.01loss_scale=1/img_gt'
            output_dir_label = './AAstanderd_imggen_joint/pon_split/imggen_full_25000_0.01loss_scale=1/label'

            file_name_imgen = os.path.join(output_dir_imgen, f"imggen_{step * 8 + i}.jpg")
            file_name_bev_cond = os.path.join(output_dir_bev_cond, f"cond_bev_{step * 8 + i}.jpg")
            file_name_gen_project = os.path.join(output_dir_gen_project, f"gen_project_{step * 8 + i}.jpg")
            file_name_imggt = os.path.join(output_dir_imggt, f"gt_img_{step * 8 + i}.jpg")
            file_name_label = os.path.join(output_dir_label, f"label_{step * 8 + i}.png")


            
            Image.fromarray(labels.astype(np.int32), mode='I').save(file_name_label)



            img_gen.save(file_name_imgen)
            bev_cond.save(file_name_bev_cond)
            bev_project.save(file_name_gen_project)
            img_gt.save(file_name_imggt)

            # visualize_map_mask(pred_img,file_name)
            # visualize_map_mask(gt_img,file_name_origin)
            
            
            print("saved a batch results")



         

    #         pred_tensor = torch.from_numpy(pred).float().unsqueeze(0)  # 假设 pred 是 float 类型
    #         gt_tensor = torch.from_numpy(gt).float().unsqueeze(0)
    #         gt_mask_tensor = torch.from_numpy(gt_mask).float().unsqueeze(0)
    #         gt_mask_tensor = gt_mask_tensor.repeat(1, 1, 1, 1)

            # processed_samples += 1


    #         for key in confusion.keys():
    #             confusion[key].update(pred_tensor > float(key), gt_tensor > 0, gt_mask_tensor > 0.5)

        main_pbar.update(8)
        main_pbar.set_postfix({"completed": f"{main_pbar.n}/{total_samples}"})
      
    # main_pbar.close()

    # for key in confusion:
    #     print(key,file=log_file)
    #     print(confusion[key].iou,file=log_file)
    #     print(confusion[key].mean_iou,file=log_file)
    #     print('double_updown_cross_45000',file=log_file)
