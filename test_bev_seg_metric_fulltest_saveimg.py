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

from accelerate import DistributedDataParallelKwargs
from utils.confusion import BinaryConfusionMatrix
from utils.utils import get_prompt
from unet2d.unet import UNet2D , get_feature_dic, clear_feature_dic

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
        num_workers=8,
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():

    max_pooling = torch.nn.MaxPool2d(kernel_size=10, stride=1, padding=1)
    pred_resize = torch.nn.Upsample(size=(256, 256), mode='nearest')


    model_configs = load_model_configs()
    cur_model = "depth"
    pipeline_class = StableDiffusionUniConPipeline
    pipe = load_unicon(pipeline_class, model_configs[cur_model])    
    total_samples = len(test_dataloader.dataset)

    log_file = open("/opt/data/private/hwj_autodrive/UniCon/eval_metric_visual/AAAponsplit/test.txt", "w")

    sample_count = 0
    MAX_SAMPLES = 6000
   
    
    # 创建主进度条（样本级别）
    main_pbar = tqdm(
        total=total_samples, 
        desc="Processing samples", 
        position=0,
        file=log_file  # 将进度条输出重定向到文件
    )

    

    for step, batch in enumerate(test_dataloader):
        if sample_count >= MAX_SAMPLES:
            break
       
        

        for i in range(batch["image"].shape[0]): 
            clear_feature_dic()

            if sample_count >= MAX_SAMPLES:
                break
            sample_count += 1
            height = 256
            width = 256
            # image_input = Image.open(image_input).convert("RGB")
            image_input = batch["image"][i].permute(2,0,1)
            

            camera_intrinsic = batch["intrinsics_origin"].to(device)
            camera_intrinsic = camera_intrinsic[i].unsqueeze(0).half()


            
            # 对应文本获取

            c = batch["segmentation"][i][:,:,:6]
            truck_count = batch['truck_count'][i]
            car_count = batch['car_count'][i]

            x_caption , y_caption = get_prompt(c,car_count.unsqueeze(0),car_count.unsqueeze(0))
        
            
            x_prompt = x_caption[0]
            y_prompt = y_caption[0]

            
            negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

           

            # prompts = [x_prompt,y_prompt]





            
            cond_input = torch.zeros_like(image_input)

            # cond_input = Image.new('RGB', (width, height), (255, 255, 255))
        
            # cond_input = Image.open(cond_input)
            image_mask = Image.new('RGB', (width, height), (255, 255, 255))
            cond_mask = Image.new('RGB', (width, height), (255, 255, 255))

            init_images = [image_input,  cond_input]  #C, H, W]

            masks = [image_mask] + [cond_mask] 
            
            masks = masks

            image_noise_strength = 0
            cond_noise_strength = 1.0

            sample_schedule = [[(1 - image_noise_strength, 1), (1 - cond_noise_strength, 1), 0]]

            batch_size = 1

            num_input = batch_size * 2

            joint_scale = 1.0

            num_inference_step = 25

            seed = 1230

            guidance_scale = 1.5

            cond_guidance_scale = 4

            inputs_config = {
                "num_input": num_input,
                "schedules": sample_schedule,
                "pairs": [(j, j+batch_size, joint_scale, joint_scale, cur_model) for j in range(batch_size)]
            }
            debug = False
            set_unicon_config_inference(pipe.unet, inputs_config["pairs"], batch_size = inputs_config["num_input"], debug=debug)
            load_scheduler(pipe, mode = "ead")

            generator = torch.Generator(device="cuda")
            generator = generator.manual_seed(seed)

            trigger_word = "depth"
            prompt = 'none'
            prompts = [prompt] * num_input
            # negative_prompt  = 'none'
            if negative_prompt is not None:
                negative_prompts = [negative_prompt] * num_input
            
            sample_schedule = parse_schedule(inputs_config["schedules"], num_inference_step)
            enable_joint_attn = True
            patch.set_joint_attention(pipe.unet, enable = enable_joint_attn)

            # print('prompts',prompts)
            # print('negative_prompts',negative_prompts)



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
                "camera_intrinsic":camera_intrinsic,
            }


            with torch.autocast(device_type = "cuda", dtype = torch.float16):
                images = pipe(**inference_config).images

            split_tensors = torch.split(images, split_size_or_sections=1, dim=0)
            x_images = split_tensors[0]
            y_images = split_tensors[1]
            bev_pred , bev_project, labels = process_segmentation(y_images[0])
            img_gt = process_rgb(image_input)

            y_tensor = ((y_images + 1) / 2).clip(0, 1)
            # y_tensor = max_pooling(y_tensor)
            y_tensor = (y_tensor > 0.5)
            
            gt_all = batch["segmentation"][i].permute(2,0,1).to(device).detach()
            gt_all = ((gt_all + 1) / 2).clip(0, 1)
            gt_mask = gt_all[-1:, :, :].unsqueeze(dim=0)
            gt = gt_all[:6, :, :].unsqueeze(dim=0)

          
            # 缩放与池化膨胀
            # gt = pred_resize(gt)
            
            # y_tensor = pred_resize(y_tensor)
            # gt_mask = pred_resize(gt_mask)
            

            # pred = (y_tensor > 0.5).float().squeeze().cpu().numpy()
            pred = y_tensor.float().squeeze().cpu().numpy()
            print("pred.shape",pred.shape)
            gt = (gt > 0).float().squeeze().cpu().numpy()

            gt_mask = 1 - gt_mask
            gt_mask = (gt_mask > 0.5).float().squeeze().cpu().numpy()
            mask_expanded = gt_mask[np.newaxis, :, :]  # 形状变为 (1, H, W)
            # # 将掩码复制到每个通道

             
          

            mask_expanded = np.repeat(mask_expanded, pred.shape[0], axis=0)
            masked_pred = np.where(mask_expanded > 0, pred, 0)

            masked_bev = np.where(mask_expanded > 0, gt, 0)
            masked_pred = np.fliplr(masked_pred)
            masked_bev = np.fliplr(masked_bev)
            masked_pred = viz_bev(masked_pred).pil
            masked_bev = viz_bev(masked_bev).pil

            # /opt/data/private/hwj_autodrive/UniCon/eval_metric_visual/AAAbev_cond_imgen_visualize/finnal_mdoel_small_visual

            output_dir_masked_pred = '/opt/data/private/hwj_autodrive/UniCon/eval_metric_visual/AAAbev_cond_imgen_visualize/finnal_model_small_visual_extime/pred'
            output_dir_masked_bev = '/opt/data/private/hwj_autodrive/UniCon/eval_metric_visual/AAAbev_cond_imgen_visualize/finnal_model_small_visual_extime/bev_gt'
            output_dir_img = '/opt/data/private/hwj_autodrive/UniCon/eval_metric_visual/AAAbev_cond_imgen_visualize/finnal_model_small_visual_extime/img_gt'
            output_dir_label = '/opt/data/private/hwj_autodrive/UniCon/eval_metric_visual/AAAbev_cond_imgen_visualize/finnal_model_small_visual_extime/label'
            output_dir_project = '/opt/data/private/hwj_autodrive/UniCon/eval_metric_visual/AAAbev_cond_imgen_visualize/finnal_model_small_visual_extime/gen_project'
            

            file_name_masked_pred = os.path.join(output_dir_masked_pred, f"pred_{step * 8 + i}.jpg")
            file_name_masked_bev = os.path.join(output_dir_masked_bev, f"bev_gt_{step * 8 + i}.jpg")
            file_name_img  = os.path.join(output_dir_img, f"img_gt_{step * 8 + i}.jpg")
            file_name_label = os.path.join(output_dir_label, f"label_{step * 8 + i}.png")
            file_name_project = os.path.join(output_dir_project, f"gen_project_{step * 8 + i}.jpg")
           
            # masked_pred.save(file_name_masked_pred)
            # masked_bev.save(file_name_masked_bev)
            Image.fromarray(labels.astype(np.int32), mode='I').save(file_name_label)
            masked_bev.save(file_name_masked_bev)
            bev_pred.save(file_name_masked_pred)
            bev_project.save(file_name_project)
            img_gt.save(file_name_img)

    #         print("saved a batch results cond seggg")
    #         if sample_count == total_samples:
    #             print("\n" + "="*50)
    #             print("LAST SAMPLE INFERENCE CONFIGURATION")
    #             print("="*50)
    #             for key, value in inference_config.items():
    #                 # 特殊处理嵌套结构
    #                 if key == "sample_schedule":
    #                     # 假设是二维列表/元组结构
    #                     print(f'    "{key}" : [',file=log_file)
    #                     for subitem in value:
    #                         print(f'        {subitem},',file=log_file)
    #                     print(f'    ],')
    #                 elif key == "generator":
    #                     print(f'    "{key}" : {type(value).__name__}(),',file=log_file) 
    #                 else:
    #                     # 对于其他值，直接打印表示形式
    #                     print(f'    "{key}" : {repr(value)},',file=log_file)
    #             print("}")
                    
               
                
                
    #             print("="*50)
            



         

    #         pred_tensor = torch.from_numpy(pred).float().unsqueeze(0)  # 假设 pred 是 float 类型
    #         gt_tensor = torch.from_numpy(gt).float().unsqueeze(0)
    #         gt_mask_tensor = torch.from_numpy(gt_mask).float().unsqueeze(0)
    #         gt_mask_tensor = gt_mask_tensor.repeat(1, 1, 1, 1)

            


    #         for key in confusion.keys():
    #             confusion[key].update(pred_tensor > float(key), gt_tensor > 0, gt_mask_tensor > 0.5)

    #     main_pbar.update(8)
    #     main_pbar.set_postfix({"completed": f"{main_pbar.n}/{total_samples}"})
      
    # main_pbar.close()

    # for key in confusion:
    #     print(key,file=log_file)
    #     print(confusion[key].iou,file=log_file)
    #     print(confusion[key].mean_iou,file=log_file)
    #     print('double_updown_cross_45000',file=log_file)
    #     print(log_file.name)
     
    

  
