import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
import os
import torch.nn as nn
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from utils.utils import blip2_cap, get_combined_filename, save_png_with_comment, set_unicon_config_inference, \
    parse_schedule, process_images
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AsymmetricAutoencoderKL, AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers import PNDMScheduler, StableDiffusionPipeline
from diffusers import DDIMScheduler

from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from utils.utils import get_active_adapters
from patch import patch
import copy
import pdb
from PIL import Image
from multi_view_generation.bev_utils import viz_bev
from multi_view_generation.bev_utils import bev_pixels_to_cam , render_map_in_image, cam_pixels_to_bev

def modify_unet_for_16_channels(unet):
    """修改UNet2DConditionModel以支持16通道输入输出"""
    # 1. 保存原始配置
    original_config = unet.config

    # 2. 修改输入卷积层 (conv_in)
    # 原始输入层权重形状: [320, 4, 3, 3]
    original_conv_in_weight = unet.conv_in.weight.data
    original_conv_in_bias = unet.conv_in.bias.data

    # 创建新的16通道输入层
    new_conv_in = nn.Conv2d(
        16,  # 新输入通道数
        original_conv_in_weight.shape[0],  # 输出通道数不变
        kernel_size=3,
        padding=1
    )

    unet.conv_in = new_conv_in

    # 3. 修改输出卷积层 (conv_out)
    # 原始输出层权重形状: [4, 320, 3, 3]
    original_conv_out_weight = unet.conv_out.weight.data
    original_conv_out_bias = unet.conv_out.bias.data

    # 创建新的16通道输出层
    new_conv_out = nn.Conv2d(
        original_conv_out_weight.shape[1],  # 输入通道数不变
        16,  # 新输出通道数
        kernel_size=3,
        padding=1
    )

    unet.conv_out = new_conv_out

    # 4. 更新模型配置
    unet.config.in_channels = 16
    unet.config.out_channels = 16

    return unet


def load_unicon_weights(unet, checkpoint_path, post_joint, model_name=None,
                        adapter_names=["xy_lora", "yx_lora", "y_lora"]):
    active_adapters = []

    for lora_name in adapter_names:
        save_dir = os.path.join(checkpoint_path, lora_name)
        if os.path.exists(save_dir):
            lora_state_dict, lora_network_alphas = StableDiffusionPipeline.lora_state_dict(save_dir)
            if model_name is not None:
                adapter_name = f"{model_name}_{lora_name}"
            else:
                adapter_name = lora_name
            StableDiffusionPipeline.load_lora_into_unet(lora_state_dict, lora_network_alphas, unet=unet,
                                                        adapter_name=adapter_name)
            active_adapters.append(adapter_name)

    model_path = os.path.join(checkpoint_path, "model.pth")
    assert os.path.exists(model_path), f"{model_path} is not found."

    patch.add_post_joint(unet, model_name, post=post_joint)

    state_dict = torch.load(model_path, map_location="cpu")

    state_dict = {key: value for key, value in state_dict.items() if 'lora' not in key}
    # Deal with training weight name
    print("\n重命名后的权重键名:")
    print("\n".join(state_dict.keys()))
    state_dict = {key.replace("conv1n", f"post1n.{model_name}"): value for key, value in state_dict.items()}

    missing_keys, unexpected_keys = unet.load_state_dict(state_dict, strict=False)

    if hasattr(unet, "unicon_adapters"):
        unet.unicon_adapters[model_name] = active_adapters
    else:
        unet.unicon_adapters = {model_name: active_adapters}

    print(model_name, "model weights loaded")

    return active_adapters


def process_segmentation(y_img_pil):
    E01 = np.array([
        [5.6847786e-03, -9.9998349e-01, 8.0507132e-04, 5.0603189e-03],
        [-5.6366678e-03, -8.3711528e-04, -9.9998379e-01, 1.5205332e+00],
        [9.9996793e-01, 5.6801485e-03, -5.6413338e-03, -1.6923035e+00],
        [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]
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
    y_tensor = ((y_img_pil + 1) / 2).clip(0, 1).permute(1, 2, 0).cpu().numpy()
    y_tensor = (y_tensor > 0.5)
    channel_coords = bev_pixels_to_cam(y_tensor, E01)
    bev_cam = render_map_in_image(intrinsics01, channel_coords)
    bev_cam_imge = viz_bev(bev_cam)
    colored_seg = viz_bev(y_tensor)  # 假设使用 nuScenes 数据集
    return colored_seg.pil, bev_cam_imge.pil


def process_rgb(x_img_pil):
    x_tensor = ((x_img_pil + 1) / 2).clip(0, 1).permute(1, 2, 0).cpu().numpy()
    x_tensor = (x_tensor * 255).astype(np.uint8)

    # 创建PIL图像
    pil_img = Image.fromarray(x_tensor, 'RGB' if x_tensor.shape[-1] == 3 else 'L')

    return pil_img

def ensure_lora_mask(unet):
    """确保所有层都有lora_mask属性"""
    for name, module in unet.named_modules():
        # 只处理需要LoRA的线性层
        if isinstance(module, nn.Linear) and not hasattr(module, 'lora_mask'):
            module.lora_mask = nn.ModuleDict()
    return unet




if __name__ == '__main__':
    vae_12c_config = {
        "in_channels": 12,
        "out_channels": 12,
        "block_out_channels": [128, 256, 512, 512],
        "layers_per_block": 2,
        "latent_channels": 16,
        "norm_num_groups": 32,
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        "act_fn": "silu",
    }

    #noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    print("加载PNDM调度器...")
    scheduler = PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    print("加载tokenizer ...")
    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="tokenizer", revision=None
    )
    print("加载 text_encoder...")
    text_encoder = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="text_encoder",  revision=None
    )

    print("加载原始vae...")
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae",  revision=None, variant=None
    )

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_12c = AutoencoderKL(**vae_12c_config).to(device)
    pretrained_path = '/opt/data/private/hwj_autodrive/UniCon/vae_output/full_sd_vae/checkpoint_epoch_50.pth'

    # 加载权重到已有模型（关键步骤）
    checkpoint = torch.load(pretrained_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # 创建新的状态字典，移除"vae."前缀
    new_state_dict = {k.replace('vae.', ''): v for k, v in state_dict.items()}

    print("加载12c_vae...")
    vae_12c.load_state_dict(new_state_dict)

    # 设置为评估模式
    vae_12c.eval()

    # unet加载
    unet_class = UNet2DConditionModel

    unet = unet_class.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet", revision=None, variant=None
    )

    unet = modify_unet_for_16_channels(unet)

    # unet = unet.to(device)
    vae = vae.to(device)
    unet.base_model_id = "runwayml/stable-diffusion-v1-5"
    
    patch.apply_patch(unet)
    unet = ensure_lora_mask(unet)
    post_joint = "conv_fuse"
    checkpoint_path = "weights/depth"
    model_name = "depth"
    # {"model_name": "depth", "base_model_id": "runwayml/stable-diffusion-v1-5", "checkpoint_path": "weights/depth",
    #  "post_joint": "conv_fuse", "trigger_word": "depth_map, ", "adapter_names": ["xy_lora", "yx_lora", "y_lora"]}
    patch.initialize_joint_layers(unet, post=post_joint)
    print("UniCon layers initialized.")

    

    active_adapters = load_unicon_weights(unet, checkpoint_path, post_joint, model_name=model_name)

   

    unet.set_adapters(active_adapters)
    patch.hack_lora_forward(unet)
    print("load_unicon_weights over")

    active_adapters = get_active_adapters(unet)

    unet.set_adapters(active_adapters)

   
    

    


    def unconditional_generation_pndm(
            unet,
            scheduler: PNDMScheduler,    # 必须添加
            num_inference_steps: int = 50,  # PNDM通常需要更多步数
            height: int = 256,
            width: int = 256,
            seed: int = None,
            output_path: str = "unconditional_pndm.png"
    ) -> Image.Image:
        """使用原始PNDM调度器的无条件图像生成"""
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)

        print(f"生成设置: {height}x{width} 图像, {num_inference_steps} 步, 种子={seed}")

        batch_size = 1
        joint_scale = 1

        generator = torch.Generator(device="cuda")
        generator = generator.manual_seed(seed)
        inputs_config = {
            "num_input": batch_size,
            'pairs' :[(0, batch_size, joint_scale, joint_scale, "depth")  # 单个配对
]
        }
        
        set_unicon_config_inference(unet, inputs_config["pairs"], batch_size=inputs_config["num_input"],
                                    debug=False ,use_cfg =False )

        # 1. 创建零张量文本嵌入（兼容模型输入）
        text_embedding_dim = unet.config.cross_attention_dim
        null_embeddings = torch.zeros(2, 77, 768).to(device)
        # torch.randn(batch_zero, 77, 768)

        # 2. 初始化潜在空间
        latents = torch.randn(
            (2, unet.config.in_channels, height // 8, width // 8),
            device=device,
            generator=torch.Generator(device=device).manual_seed(seed) if seed else None
        )

        assert unet.config.in_channels == 16, "通道数必须为16"


        print('latents.shape',latents.shape)

        # 3. 设置调度器
        scheduler.set_timesteps(num_inference_steps)
        latents = latents * scheduler.init_noise_sigma

          # 复制原始 latent（避免原地修改原始数据）
        processed_latents = latents.clone()

        # 对第一个批次（索引0）的后12个通道（索引4到15）置零
        processed_latents[0, 4:] = 0  # 更安全的切片操作

        latents = processed_latents

        print("开始去噪过程...")
        patch.set_joint_attention(unet, enable = True)
        unet = unet.to(device)
        

        # 4. PNDM迭代去噪过程
        for i, t in enumerate(scheduler.timesteps):
            # 预测噪声残差
            with torch.no_grad():
                noise_pred = unet(
                    latents,
                    t,
                    encoder_hidden_states=null_embeddings  # 传入空嵌入
                ).sample

            # 使用PNDM调度器更新潜变量
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            # 进度更新
            if (i + 1) % 10 == 0 or (i + 1) == num_inference_steps:
                print(f"  步骤 {i + 1}/{num_inference_steps} 完成")

        # 5. 解码潜变量为图像
        print("解码图像...")
        latents = 1 / vae.config.scaling_factor * latents
        with torch.no_grad():
            split_tensors = torch.split(latents, split_size_or_sections=1, dim=0)
            rgb_latent = split_tensors[0]
            bev_latent = split_tensors[1]
            rgb_latent_valid = rgb_latent[:, :4, :, :]
            image_split = vae.decode(rgb_latent_valid).sample

            tensor1, tensor2, tensor3, tensor4 = torch.split(bev_latent, split_size_or_sections=4, dim=1)
              
            bev_split1 = vae.decode(tensor1).sample
            bev_split2 = vae.decode(tensor2).sample
            bev_split3 = vae.decode(tensor3).sample
            bev_split4 = vae.decode(tensor4).sample
                

            bev_split = torch.cat([bev_split1, bev_split2, bev_split3, bev_split4], dim=1)

            # bev_split = vae_12c.decode(bev_latent).sample

           

        # 6. 转换为PIL图像
        image_rgb = process_rgb(image_split[0])
        image_bev , bev_project = process_segmentation(bev_split[0])
        

        save_dir = "/opt/data/private/hwj_autodrive/UniCon/new_inference_results"
        os.makedirs(save_dir, exist_ok=True)
        output_path_rgb = os.path.join(save_dir, "image.png")
        output_path_bev = os.path.join(save_dir, "bev.png")
        output_path_project = os.path.join(save_dir, "project.png")
        # 保存图像
        image_rgb.save(output_path_rgb)
        image_bev.save(output_path_bev)
        bev_project.save(output_path_project)
        print(f"图像已保存至 {os.path.abspath(save_dir)}")




    ####################################

    # 开始生成图像
    image = unconditional_generation_pndm(
        unet = unet,
        scheduler = scheduler,
        num_inference_steps=50,  # PNDM通常需要50-100步
        height=256,
        width=256,
        seed=70,  # 固定种子保证可复现性
        output_path="pndm_unconditional.png"
    )


  





