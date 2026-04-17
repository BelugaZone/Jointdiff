#!/usr/bin/env python
# coding=utf-8
# Adapted from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import sys
_JOINTDIFF_ROOT = Path(__file__).resolve().parents[2]
if str(_JOINTDIFF_ROOT) not in sys.path:
    sys.path.insert(0, str(_JOINTDIFF_ROOT))

import argparse
import torch.nn as nn
import logging
import math
import os
import random
import shutil
import pdb

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from packaging import version
from peft import LoraConfig
import torchvision.transforms.v2 as transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import random 

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from patch import patch
from utils.utils import set_jointdiff_config_train
from utils.load_utils import load_image_folder
from utils.peft_utils import get_peft_model_state_dict, set_adapters_requires_grad
from safetensors import safe_open
from multi_view_generation.bev_utils.cityscapes_dataset import CityscapesDataset
from multi_view_generation.bev_utils.nuscenes_dataset import NuScenesDataset
from multi_view_generation.bev_utils import viz_bev
from multi_view_generation.bev_utils import bev_pixels_to_cam , render_map_in_image, cam_pixels_to_bev
from unet2d.unet_double_loss import UNet2D , get_feature_dic, clear_feature_dic
from seghead.segdecoder_double_loss_LSS import seg_decorder
import itertools
# from patch.share_timesteps import current_timestep , set_current_timestep, set_current_camera_intrinsic, camera_intrinsic, set_current_E, camera_E



# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.28.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="jointdiff training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--x_image_column",
        type=str,
        nargs='+',
        default=None,
        help=(
            "The column of the dataset containing x image."
        ),
    )
    parser.add_argument(
        "--y_image_column",
        type=str,
        default="image",
        nargs='+',
        help="The column of the dataset containing y image."
    )
    parser.add_argument(
        "--x_caption_column",
        type=str,
        default="text",
        nargs='+',
        help="The column of the dataset containing x caption.",
    )
    parser.add_argument(
        "--y_caption_column",
        type=str,
        nargs='+',
        default=None,
        help="The column of the dataset containing y caption. If not set, use the same as x_caption_column.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="jointdiff",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=5.0,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
             "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=32,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=100,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    # jointdiff specific arguments
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="json"
    )
    parser.add_argument(
        "--prompt_dropout_prob",
        type=float,
        default=0.1,
        help=("The prob of dropping prompt when training."),
    )
    parser.add_argument(
        "--joint_dropout_prob",
        type=float,
        default=0.0,
        help=("The prob of dropping joint modules when training."),
    )
    parser.add_argument(
        "--trigger_word",
        type=str,
        default="",
        help=("Trigger word to prepend to the y captions."),
    )
    parser.add_argument(
        "--train_y_lora",
        action="store_true",
        help=("Whether to add and train y lora."),
    )
    parser.add_argument(
        "--ylora_rank",
        type=int,
        default=None,
        help=("The dimension of the y LoRA update matrices. If not set, use the same as args.rank."),
    )
    parser.add_argument(
        "--skip_encoder",
        action="store_true",
        help=("Whether to skip the encoder when adding joint modules."),
    )
    parser.add_argument(
        "--post_joint",
        type=str,
        default="conv",
        help=("The post joint module type. Choose between 'conv' or 'conv_fuse'."),
    )
    parser.add_argument(
        "--rand_transform",
        action="store_true",
        help=("Whether to use random hw ratio for training."),
    )
    parser.add_argument(
        "--separate_xy_trans",
        action="store_true",
        help=("Whether to use separate transforms for x and y images."),
    )
    parser.add_argument(
        "--hw_logratio_max",
        type=float,
        default=1.5,
        help=(
            "The max hw logratio for random hw ratio. The ratio is sampled from [1/hw_logratio_max, hw_logratio_max]."
        ),
    )
    parser.add_argument(
        "--resume_from_model", type=str, default=None,
        help=(
            "The path to the model to resume from. The model should be a state dict of the unet in safetensor. "
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class ChannelMaskGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask):
        ctx.save_for_backward(mask)
        return input * mask

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output * mask, None

def process_segmentation(y_img_pil):
            E01 = np.array([
               [ 5.6847786e-03, -9.9998349e-01,  8.0507132e-04,  5.0603189e-03],
               [-5.6366678e-03, -8.3711528e-04, -9.9998379e-01,  1.5205332e+00],
               [ 9.9996793e-01,  5.6801485e-03, -5.6413338e-03, -1.6923035e+00],
               [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
                ])
            
            # intrinsics01 = np.array([
            #            [202.62675249, 0.0, 130.60272316],
            #            [0.0, 360.22533776, 139.80645427],
            #            [0.0, 0.0, 1.0]
            #            ])
            intrinsics01 = np.array([
                       [405.26,   0.0,         261.2],
                       [  0.0,         720.44, 279.62],
                       [  0.0,           0.0,           1.0        ]
                       ])
            y_tensor = ((y_img_pil + 1) / 2).clip(0, 1).permute(1,2,0).cpu().numpy()
            y_tensor = (y_tensor > 0.5)
            channel_coords = bev_pixels_to_cam(y_tensor, E01)
            bev_cam = render_map_in_image(intrinsics01, channel_coords)
            bev_cam_imge = viz_bev(bev_cam)
            colored_seg = viz_bev(y_tensor)  # 假设使用 nuScenes 数据集
            return colored_seg.pil , bev_cam_imge.pil

def process_rgb(x_img_pil):
            x_tensor = ((x_img_pil + 1) / 2).clip(0, 1).permute(1,2,0).cpu().numpy()
            x_tensor = (x_tensor * 255).astype(np.uint8)
    
             # 创建PIL图像
            pil_img = Image.fromarray(x_tensor, 'RGB' if x_tensor.shape[-1] == 3 else 'L')
            
            return pil_img

def get_ddpm_params(scheduler, timesteps):
    # 获取完整序列（确保在正确设备上）
    alphas = scheduler.alphas.to(timesteps.device)
    alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)
    betas = scheduler.betas.to(timesteps.device)
    
    # 获取批处理的时间步相关值
    alpha_t = alphas.gather(0, timesteps)          # [batch_size]
    alpha_cumprod_t = alphas_cumprod.gather(0, timesteps) # [batch_size]
    beta_t = betas.gather(0, timesteps)            # [batch_size]
    
    # 计算 sigma_t = √(1 - alpha_cumprod_t)
    sigma_t = (1 - alpha_cumprod_t).sqrt()         # [batch_size]
    
    # 重塑为适合噪声添加的形式 [batch_size, 1, 1, 1]
    shape_t = (timesteps.shape[0], 1, 1, 1)
    alpha_t = alpha_t.view(shape_t)
    alpha_cumprod_t = alpha_cumprod_t.view(shape_t)
    beta_t = beta_t.view(shape_t)
    sigma_t = sigma_t.view(shape_t)
    
    return alpha_t, alpha_cumprod_t, beta_t, sigma_t

def compute_sqrt_alpha_bar(scheduler, t):
    # 获取调度器的噪声累积参数表
    alphas_cumprod = scheduler.alphas_cumprod
    
    # 扩展维度以支持批处理
    batch_size = t.shape[0]
    half = batch_size // 2
    sqrt_alpha_bar = alphas_cumprod[t].view(batch_size, 1, 1, 1)
    sqrt_alpha_bar_bev2img = sqrt_alpha_bar[half:,:,:,:]
    sqrt_alpha_bar_img2bev = sqrt_alpha_bar[:half,:,:,:]

    

    # 计算平方根
    return torch.sqrt(sqrt_alpha_bar_bev2img) , torch.sqrt(sqrt_alpha_bar_img2bev)


def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
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

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_12c = AutoencoderKL(**vae_12c_config).to(device)

    pretrained_path = '/opt/data/private/hwj_autodrive/jointdiff/vae_output/full_sd_vae/checkpoint_epoch_50.pth'

    # 加载权重到已有模型
    checkpoint = torch.load(pretrained_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # 创建新的状态字典，移除"vae."前缀
    new_state_dict = {k.replace('vae.', ''): v for k, v in state_dict.items()}

    vae_12c.load_state_dict(new_state_dict)

    unet_class = UNet2D

    unet = unet_class.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    seg_head = seg_decorder((256, 256)).cuda()

    def modify_unet_for_16_channels(unet: UNet2DConditionModel):
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

        # 初始化新权重：前4个通道使用预训练权重，后12个通道使用高斯初始化
        with torch.no_grad():
            # 复制前4个通道的权重
            new_conv_in.weight[:, :4] = original_conv_in_weight

            # 初始化后12个通道的权重
            # 使用与原始权重相同的标准差进行初始化
            std = original_conv_in_weight.std().item()
            new_conv_in.weight[:, 4:] = torch.randn_like(new_conv_in.weight[:, 4:]) * std * 0.1

            # 偏置保持不变
            new_conv_in.bias.data = original_conv_in_bias

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

        # 初始化新权重：前4个通道使用预训练权重，后12个通道使用高斯初始化
        with torch.no_grad():
            # 复制前4个输出通道的权重
            new_conv_out.weight[:4] = original_conv_out_weight

            # 初始化后12个输出通道的权重
            # 使用与原始权重相同的标准差进行初始化
            std = original_conv_out_weight.std().item()
            new_conv_out.weight[4:] = torch.randn_like(new_conv_out.weight[4:]) * std * 0.1

            # 偏置：前4个通道使用原始偏置，后12个初始化为0
            new_conv_out.bias.data[:4] = original_conv_out_bias
            new_conv_out.bias.data[4:] = 0

        unet.conv_out = new_conv_out

        # 4. 更新模型配置
        unet.config.in_channels = 16
        unet.config.out_channels = 16

        return unet

    def modify_unet_for_8_channels(unet):
        """修改UNet2DConditionModel以支持8通道输入输出"""
        # 1. 保存原始配置
        original_config = unet.config

        # 2. 修改输入卷积层 (conv_in)
        # 原始输入层权重形状: [320, 4, 3, 3]
        original_conv_in_weight = unet.conv_in.weight.data
        original_conv_in_bias = unet.conv_in.bias.data

        # 创建新的8通道输入层
        new_conv_in = nn.Conv2d(
            8,  # 新输入通道数 (4->8)
            original_conv_in_weight.shape[0],  # 输出通道数保持320
            kernel_size=3,
            padding=1
        )

        # 初始化新权重：前4个通道和后4个通道都使用预训练权重
        with torch.no_grad():
            # 复制前4个通道的权重
            new_conv_in.weight[:, :4] = original_conv_in_weight

            # 将原始权重复制一份给后4个通道
            new_conv_in.weight[:, 4:] = original_conv_in_weight

            # 偏置保持不变
            new_conv_in.bias.data = original_conv_in_bias

        unet.conv_in = new_conv_in

        # 3. 修改输出卷积层 (conv_out)
        # 原始输出层权重形状: [4, 320, 3, 3]
        original_conv_out_weight = unet.conv_out.weight.data
        original_conv_out_bias = unet.conv_out.bias.data

        # 创建新的8通道输出层
        new_conv_out = nn.Conv2d(
            original_conv_out_weight.shape[1],  # 输入通道数保持320
            8,  # 新输出通道数 (4->8)
            kernel_size=3,
            padding=1
        )

        # 初始化新权重：前4个输出通道和后4个输出通道都使用预训练权重
        with torch.no_grad():
            # 复制前4个输出通道的权重
            new_conv_out.weight[:4] = original_conv_out_weight

            # 将原始权重复制一份给后4个输出通道
            new_conv_out.weight[4:] = original_conv_out_weight

            # 偏置：前4个通道使用原始偏置，后4个通道也复制相同的偏置
            new_conv_out.bias.data[:4] = original_conv_out_bias
            new_conv_out.bias.data[4:] = original_conv_out_bias

        unet.conv_out = new_conv_out

        # 4. 更新模型配置
        unet.config.in_channels = 8
        unet.config.out_channels = 8

        return unet

    def classify_unet_params(unet):
        """更健壮的参数分类函数"""
        params_dict = {
            "new_input_layer": [],
            "new_output_layer": [],
            "original_lora": [],
            "frozen": []
        }

        # 收集新输入层名称 (不存储张量)
        new_input_names = []
        if unet.conv_in.weight.shape[1] > 4:
            new_input_names.append("conv_in.weight")
            if hasattr(unet.conv_in, "bias") and unet.conv_in.bias is not None:
                new_input_names.append("conv_in.bias")

        # 收集新输出层名称
        new_output_names = []
        if unet.conv_out.weight.shape[0] > 4:
            new_output_names.append("conv_out.weight")
            if hasattr(unet.conv_out, "bias") and unet.conv_out.bias is not None:
                new_output_names.append("conv_out.bias")

        # 分类所有参数
        for name, param in unet.named_parameters():
            # 使用名称匹配而非张量匹配
            if name in new_input_names:
                params_dict["new_input_layer"].append(param)
                print(f"new_input_layer: {name}")  # 输出参数名称而不是整个列表
            elif name in new_output_names:
                params_dict["new_output_layer"].append(param)
                print(f"new_output_layer: {name}")  # 输出参数名称而不是整个列表
            # elif y_lora_config:
            #     if any(module in name for module in y_lora_config.target_modules):
            #           params_dict["original_lora"].append(param)
            else:
                params_dict["frozen"].append(param)

        return params_dict

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    ignore_keys = ["load_maximally", "cond_stage_model"]
    unet = modify_unet_for_8_channels(unet)



    # unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae_12c.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    seg_head.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    vae_12c.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Freeze the unet parameters before adding adapters

    # param_groups = classify_unet_params(unet)
    # for param in param_groups["frozen"]:
    #     param.requires_grad = False

    # for param in unet.parameters():
    #     param.requires_grad_(False)

    # Insert and initialize jointdiff modules
    patch.apply_patch(unet, name_skip= None if args.skip_encoder else None, train=True)

    patch.initialize_joint_layers(unet, post=args.post_joint)

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["attn1n.to_k", "attn1n.to_q", "attn1n.to_v", "attn1n.to_out.0"],
    )

    all_loras = []

    # Add adapter and make sure the trainable params are in float32.

    # unet.add_adapter(unet_lora_config, adapter_name="xy_lora")
    # unet.add_adapter(unet_lora_config, adapter_name="yx_lora")
    # all_loras += ["xy_lora", "yx_lora"]

    # if args.train_y_lora:
    #     y_lora_config = LoraConfig(
    #         r=args.rank if args.ylora_rank is None else args.ylora_rank,
    #         lora_alpha=args.rank if args.ylora_rank is None else args.ylora_rank,
    #         init_lora_weights="gaussian",
    #         target_modules=["attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0", "attn2.to_k", "attn2.to_q",
    #                         "attn2.to_v", "attn2.to_out.0",  # 添加于新输入输出层相关的模块
    #                         "conv_in",  # 新输入层
    #                         "conv_out",  # 新输出层
    #                         "conv_blocks.*"],
    #     )
    #     unet.add_adapter(y_lora_config, adapter_name="y_lora")
    #     all_loras += ["y_lora"]

    # patch.hack_lora_forward(unet)

    if args.resume_from_model is not None:
        logger.info(f"Resuming from {args.resume_from_model}")
    
        # 检查文件类型
        if args.resume_from_model.endswith(".pth"):
            # 处理.pth文件的特殊逻辑
            model_path = args.resume_from_model
            assert os.path.exists(model_path), f"{model_path} not found"
            # 加载并处理权重
            state_dict = torch.load(model_path, map_location="cpu")
            state_dict = {k: v for k, v in state_dict.items() if 'lora' not in k}
            state_dict = {k.replace("conv1n", f"post1n.{model_name}"): v for k, v in state_dict.items()}
            
            # 加载权重
            unet.load_state_dict(state_dict, strict=False)
        
        else:
            # 处理其他格式（如safetensors）
            state_dict = {}
            with safe_open(args.resume_from_model, framework="pt", device="cpu") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
            unet.load_state_dict(state_dict, strict=False)

    # unet.set_adapters(all_loras)

    # param_groups = classify_unet_params(unet)
    # for param in param_groups["frozen"]:
    #     param.requires_grad = False

    # for param in unet.parameters():
    #     param.requires_grad_(False)

    trainable_loras = all_loras
    # set_adapters_requires_grad(unet, True, trainable_loras)

    patch.set_joint_layer_requires_grad(unet, ["xy_lora", "yx_lora"], True)

    new_layer_params = []
    for name, param in unet.named_parameters():
        #  if "conv_in" in name or "conv_out" in name:
            new_layer_params.append(param)
            param.requires_grad = True  # 强制设置为可训练
            # if 'attn2' in name:  # 在Diffusers实现中，attn1通常是自注意力，attn2是交叉注意力
            #     print(f"Freezing cross-attention layer: {name}")
            #     param.requires_grad_(False)

    def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
               trainable_params += param.numel()
    
        print(
            f"trainable params: {trainable_params} || "
            f"all params: {all_param} || "
            f"trainable: {100 * trainable_params / all_param:.2f}%"
         )

        # 使用方式
    print_trainable_parameters(unet)
    print_trainable_parameters(seg_head)

    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def load_model_hook(models, input_dir):

        for model in models:
            if isinstance(model, unet_class):
                unet = model

        save_path = os.path.join(input_dir, "model.pth")
        state_dict = torch.load(save_path, map_location="cpu")

        unet.load_state_dict(state_dict, strict=False)

    def save_model_hook(models, weights, output_dir):

        for model in models:
            if isinstance(model, unet_class):
                unet = model

        state_dict = dict()

        for name, params in unet.named_parameters():
            if params.requires_grad:
                state_dict[name] = params

        save_path = os.path.join(output_dir, "model.pth")
        torch.save(state_dict, save_path)

    accelerator.register_load_state_pre_hook(load_model_hook)
    accelerator.register_save_state_pre_hook(save_model_hook)

    trainable_params = filter(
            lambda p: p.requires_grad,
            itertools.chain(
                unet.parameters(),
                seg_head.parameters()
            )
        )

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

  

    def get_prompt(masks):
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

    


    


    def random_train_transforms():
        hw_logratio_range = [-np.log(args.hw_logratio_max), np.log(args.hw_logratio_max)]
        total_pixels = args.resolution * args.resolution
        hw_logratio = random.uniform(*hw_logratio_range)
        # hw_logratio = np.log(1.5)
        hw_ratio = np.e ** (hw_logratio)
        # hw_ratio =  0.5
        width = int(np.sqrt(total_pixels / hw_ratio) / 8 + 0.5) * 8
        height = int(total_pixels // width / 8 + 0.5) * 8
        # random_sizes = [(512, 512), (576, 448), (576, 384), (448, 576), (384, 576), (640, 448), (448, 640)]
        # height, width = random.choice(random_sizes)
        hw_ratio = height / width
        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=(height, width), scale=(0.75, 1.0),
                                             ratio=(1 / hw_ratio, 1 / hw_ratio)),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        return train_transforms

    # if args.rand_transform:
    #     train_transforms = random_train_transforms()

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, x_text_column, y_text_column, trigger_word=""):

        if x_text_column not in examples and "json" in examples:
            tmp_dict = dict()
            tmp_dict[x_text_column] = [exp[x_text_column] for exp in examples["json"]]
            if y_text_column is not None:
                tmp_dict[y_text_column] = [exp[y_text_column] for exp in examples["json"]]
            examples = tmp_dict

        captions = [caption for caption in examples[x_text_column]]

        if y_text_column is not None:
            y_captions = [caption for caption in examples[y_text_column]]
        else:
            y_captions = captions

        y_captions = [trigger_word + cpt for cpt in y_captions]

        captions += y_captions
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def preprocess_train(examples):
        # For possible multiple captions, we randomly select one.
        x_image_column, x_text_column = random.choice(ximage_text_columns)
        y_image_column, y_text_column = random.choice(yimage_text_columns)

        x_images = []
        y_images = []
        for x_image, y_image in zip(examples[x_image_column], examples[y_image_column]):
            # check x y sizes match
            assert args.separate_xy_trans or (x_image.width == y_image.width and x_image.height == y_image.height)

            x_images.append(x_image.convert("RGB"))
            y_images.append(y_image.convert("RGB"))

        # pad_num = len(examples[x_image_column]) - len(x_images)
        # x_images = x_images + x_images[-1:] * pad_num
        # y_images = y_images + y_images[-1:] * pad_num

        if args.separate_xy_trans:
            x_images = [train_transforms(x_image) for x_image in x_images]
            y_images = [train_transforms(y_image) for y_image in y_images]
        else:
            xy_images = [train_transforms(x_image, y_image) for x_image, y_image in zip(x_images, y_images)]
            x_images, y_images = zip(*xy_images)

        examples["x_pixel_values"] = x_images
        examples["y_pixel_values"] = y_images

        input_ids = tokenize_captions(examples, x_text_column=x_text_column, y_text_column=y_text_column,
                                      trigger_word=args.trigger_word)
        x_len = len(input_ids) // 2
        examples["x_input_ids"] = input_ids[:x_len]
        examples["y_input_ids"] = input_ids[x_len:]
        return examples

    def filter_incomplete_samples(example):

        # Check if both 'image' and 'json' fields are present in the sample
        has_image = True
        for clm in x_image_column:
            has_image = has_image and example.get(clm) is not None
        for clm in y_image_column:
            has_image = has_image and example.get(clm) is not None
        has_json = example.get("json") is not None

        # resolution_pass = max(example[x_image_column[0]].size) > args.resolution

        # Return True if both are present, False otherwise
        return has_image and has_json

    def get_high_timesteps(bsz):
        shift = 3.1582
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logits_norm = torch.randn(bsz, device=device)
        logits_norm = logits_norm * 1.0  # larger scale for more uniform sampling
        timesteps = logits_norm.sigmoid()
        timesteps = (timesteps * shift) / (1 + (shift - 1) * timesteps)
        timesteps = timesteps * 1000.0
        return timesteps

    def get_small_timesteps(bsz):
        shift = 0.25
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logits_norm = torch.randn(bsz, device=device)
        logits_norm = logits_norm * 1.0  # larger scale for more uniform sampling
        timesteps = logits_norm.sigmoid()
        timesteps = (timesteps * shift) / (1 + (shift - 1) * timesteps)
        timesteps = timesteps * 1000.0
        return timesteps

    # with accelerator.main_process_first():
    #     if args.max_train_samples is not None:
    #         dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    #     # Set the training transforms

    #     if args.dataset_type == "webdataset":
    #         dataset["train"] = dataset["train"].filter(filter_incomplete_samples)

    #     if hasattr(dataset["train"], "with_transform"):
    #         train_dataset = dataset["train"].with_transform(preprocess_train)
    #     else:
    #         train_dataset = dataset["train"].map(preprocess_train, batched=True, batch_size=args.train_batch_size)

    def collate_fn(examples):
        x_pixel_values = torch.stack([example["x_pixel_values"] for example in examples])
        x_pixel_values = x_pixel_values.to(memory_format=torch.contiguous_format).float()
        y_pixel_values = torch.stack([example["y_pixel_values"] for example in examples])
        y_pixel_values = y_pixel_values.to(memory_format=torch.contiguous_format).float()
        x_input_ids = torch.stack([example["x_input_ids"] for example in examples])
        y_input_ids = torch.stack([example["y_input_ids"] for example in examples])
        return {"x_pixel_values": x_pixel_values, "x_input_ids": x_input_ids, "y_input_ids": y_input_ids,
                "y_pixel_values": y_pixel_values}

    train_dataset_nuscenes = NuScenesDataset(
        split=0,  # 必需的位置参数
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

    train_dataset_cityscapes = CityscapesDataset(
    )
    dataset_length = len(train_dataset_nuscenes)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_nuscenes,
        shuffle=True if args.dataset_type != "webdataset" else False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    logger.info(f"Dataset length: {dataset_length}")
    logger.info(f"Dataloader length on each process {len(train_dataloader)}")

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, seg_head, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, seg_head,optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("jointdiff", config={k: v if not isinstance(v, list) else " ".join(v) for k, v in
                                                    vars(args).items()})

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {dataset_length}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    logger.info(f"Resume training from {args.resume_from_checkpoint}")

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    logger.info(f"VAE scale factor: {vae_scale_factor}")

    bsz = 0

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        seg_head.train() 
        unet = unet.to(torch.float32)
        seg_head = seg_head.to(torch.float32)
        # import pdb;pdb.set_trace()

        # 初始化损失统计
        train_loss = 0.0
        train_loss_rgb = 0.0  # 新增：RGB数据损失统计
        train_loss_bev = 0.0  # 新增：BEV数据损失统计
        train_loss_perceptual = 0.0
        train_loss_perceptual_img2bev = 0.0
        
        logger.info(f"Epoch {epoch}, Step {global_step}")

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                if args.rand_transform:
                    train_transforms = random_train_transforms()
                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                device = accelerator.device
                batch["image"] = batch["image"].permute(0, 3, 1, 2).to(device)  # 图像
                batch["segmentation"] = batch["segmentation"].permute(0, 3, 1, 2).to(device)  # 分割图
                batch_zero = batch["segmentation"].shape[0] * 2
                camera_intrinsic = batch["intrinsics_origin"].to(device) 
                camera_E = batch["E"].to(device) 
                camera_intrinsic = camera_intrinsic.half() 
                camera_E = camera_E.half() 

                

                if bsz != batch["segmentation"].shape[0]:
                    bsz = 2 * batch["segmentation"].shape[0]
                    set_jointdiff_config_train(unwrap_model(unet), bsz)



                # 采样随机时间步
                # timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
                assert bsz % 2 == 0, "Batch size must be even"
                half = bsz // 2
                
                # if current_step is None:
                    # 随机选择策略
                mode = random.randint(0, 3)
                

                if mode == 0 or mode == 1:  # 原始策略：整个批次随机
                    # timesteps_rgb = torch.randint(0, noise_scheduler.config.num_train_timesteps, (half,))
                    timesteps_rgb = get_small_timesteps(half)
                    timesteps_bev = timesteps_rgb
                    timesteps = torch.cat([timesteps_rgb, timesteps_bev])
                
                
                elif mode == 2:  # 策略2：前半0，后半随机
                    timesteps_rgb = get_small_timesteps(half)
                    timesteps_bev = get_high_timesteps(half)
                    timesteps = torch.cat([timesteps_rgb, timesteps_bev])
                
                elif mode == 3:  # 策略3：前半随机，后半0
                    timesteps_rgb = get_high_timesteps(half)
                    timesteps_bev = get_small_timesteps(half)
                    timesteps = torch.cat([timesteps_rgb, timesteps_bev])

                # timesteps_front = torch.zeros(half, dtype=torch.long)
                # timesteps_back = torch.randint(0, noise_scheduler.config.num_train_timesteps, (half,))
                # timesteps = torch.cat([timesteps_front, timesteps_back])
                

              

                timesteps = timesteps.long()
                clear_feature_dic()

                


                c = batch["segmentation"]
                c = c[:,:6,:,:]
                # c = c[:,[0,1,2,8,10,11],:,:]
                

                # img = batch["image"][0]
                # save_dir_img = "/opt/data/private/hwj_autodrive/jointdiff/groundtruth/gt_img"
                # seg , project = process_segmentation(c[0])
                # image = process_rgb(img)
                # save_dir = "/opt/data/private/hwj_autodrive/jointdiff/groundtruth/gt_bev"
                # os.makedirs(save_dir, exist_ok=True)
                # os.makedirs(save_dir_img, exist_ok=True)
                # filename = os.path.join(save_dir, f"image_{step}.png")
                # filename_project = os.path.join(save_dir, f"image_project{step}.png")
                # filename_img = os.path.join(save_dir_img, f"image_gt{step}.png")
                # seg.save(filename)
                # project.save(filename_project)
                # image.save(filename_img)
                # print('train_dataset_gt saved')

                # 文本token获取
                truck_count = batch['truck_count']
                car_count = batch['car_count']

                x_caption , y_caption = get_prompt(c)

                # print('x_caption',x_caption)
                # print('y_captions',y_caption)

                
                # captions = x_caption 
                
                # y_captions = trigger_word + y_captions

                # captions += y_captions

                x_inputs = tokenizer(
                    x_caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                )

                y_inputs = tokenizer(
                    y_caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                )

                x_input_ids = x_inputs.input_ids

                y_input_ids = y_inputs.input_ids

                # input_ids_prompt =  inputs.input_ids


                # x_len = len(input_ids_prompt) // 2
                # x_input_ids = input_ids_prompt[:x_len]
                # y_input_ids = input_ids_prompt[x_len:]

                xy_input_ids = torch.cat([x_input_ids, y_input_ids], dim=0)

                







                
               
                # 处理BEV数据
                # tensor1, tensor2, tensor3, tensor4 = torch.split(c, split_size_or_sections=3, dim=1)
                tensor1, tensor2 = torch.split(c, split_size_or_sections=3, dim=1)
                c1 = vae.encode(tensor1.to(dtype=weight_dtype)).latent_dist.sample()
                c2 = vae.encode(tensor2.to(dtype=weight_dtype)).latent_dist.sample()
                # c3 = vae.encode(tensor3.to(dtype=weight_dtype)).latent_dist.sample()
                # c4 = vae.encode(tensor4.to(dtype=weight_dtype)).latent_dist.sample()
                latents_bev = torch.cat([c1, c2], dim=1)

                # latents_bev = torch.cat([c1, c2, c3, c4], dim=1)

                # latents_bev = vae_12c.encode(c.to(dtype=weight_dtype)).latent_dist.sample()
                # 处理RGB数据
                latents_rgb = vae.encode(batch["image"].to(dtype=weight_dtype)).latent_dist.sample()
                batch_size, _, height, width = latents_rgb.shape

                # 扩展RGB数据到16通道
                zeros_to_add = torch.zeros(
                    (batch_size, 4, height, width),
                    device=latents_rgb.device,
                    dtype=latents_rgb.dtype
                )
                latents_16ch = torch.cat([latents_rgb, zeros_to_add], dim=1)
                # print('latents_16ch.shape',latents_16ch.shape)
                # print('latents_bev.shape',latents_bev.shape)
                # 拼接RGB和BEV数据
                latents = torch.cat([latents_16ch, latents_bev], dim=0)

                # #####应用掩码

                original_bsz = latents_rgb.size(0)
                rgb_mask = torch.zeros(1, 8, 1, 1, device=latents_rgb.device, dtype=latents_rgb.dtype)
                rgb_mask[:, :4, :, :] = 1.0  # RGB有效通道（前4个）
                bev_mask = torch.ones(1, 8, 1, 1, device=latents_rgb.device, dtype=latents_rgb.dtype)  # BEV全部通道有效

                # 创建整个批次的掩码
                mask = torch.cat([rgb_mask] * original_bsz + [bev_mask] * original_bsz, dim=0)

                # 转换到潜在空间
                latents = latents * vae.config.scaling_factor
                timesteps = timesteps.to(latents.device)

                # 采样噪声
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
               


                

                # 添加噪声
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(xy_input_ids.to(latents.device), return_dict=False)[0]
                
                

                if args.prompt_dropout_prob > 0:
                    random_p = torch.rand(
                        bsz // 2, device=latents.device)
                    # Sample masks for the edit prompts.
                    x_prompt_mask = random_p < 2 * args.prompt_dropout_prob
                    y_prompt_mask = torch.logical_and(random_p > args.prompt_dropout_prob, random_p < 3 * args.prompt_dropout_prob)
                    prompt_mask = torch.cat([x_prompt_mask, y_prompt_mask], dim = 0)
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # print(encoder_hidden_states.shape)
                   
                    # Final text conditioning.
                    null_conditioning = torch.zeros_like(encoder_hidden_states)
                    # print('====================')
                    # print('null_conditioning.shape',null_conditioning.shape)
                    # print('====================')
                    encoder_hidden_states = torch.where(
                        prompt_mask, null_conditioning, encoder_hidden_states)
                    

                # 联合注意力dropout
                if args.joint_dropout_prob > 0:
                    if random.random() < args.joint_dropout_prob:
                        patch.set_joint_attention(unwrap_model(unet), enable=False)
                    else:
                        patch.set_joint_attention(unwrap_model(unet), enable=True)

                # 确定目标
                if args.prediction_type is not None:
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                input_data = noisy_latents

                # 应用掩码
                masked_input = ChannelMaskGrad.apply(input_data, mask)

                # 预测噪声残差
               
                model_pred = unet(masked_input, timestep=timesteps, encoder_hidden_states=encoder_hidden_states, camera_E=camera_E,camera_intrinsic=camera_intrinsic, return_dict=False)[0]
                
                # 获取多尺度特征图
                diffusion_features = get_feature_dic()
                # for feature_level, feature_list in diffusion_features.items():
                #     print(f"\nFeature Level: {feature_level}")
                    
                #     # 遍历该特征级别的每个张量
                #     for i, tensor in enumerate(feature_list):
                #         # 获取张量形状
                #         shape = tensor.shape
                        
                #         # 打印详细信息
                #         print(f"  Tensor {i+1}: Shape={shape}, Device={tensor.device}, Dtype={tensor.dtype}")

                # 分割头分割
                seg_outputs, img2bev_seg_outputs = seg_head(diffusion_features,camera_intrinsic=camera_intrinsic)
                # print('seg_outputs',seg_outputs.shape)

                seg_target = c
                seg_target = (seg_target + 1) / 2

                sqrt_alpha_bar_t, sqrt_alpha_bar_t_img2bev = compute_sqrt_alpha_bar(noise_scheduler, timesteps)


                # _, alpha_cumprod_t, _, sigma_t = get_ddpm_params(noise_scheduler, timesteps)

                # pred_latents_clean = (noisy_latents - sigma_t * model_pred) / alpha_cumprod_t.sqrt()

                # pred_latents_clean = pred_latents_clean / vae.config.scaling_factor
            
                 # 解码为图像
                

               


                

               
                # ====== 修改点1: 分别计算RGB和BEV的损失 ======
                if args.snr_gamma is None:
                    # 前半个batch是RGB数据
                    rgb_pred = model_pred[:bsz // 2]
                    rgb_target = target[:bsz // 2]
                    rgb_pred_valid = rgb_pred[:, :4, :, :]
                    rgb_target_valid = rgb_target[:, :4, :, :]
                    loss_rgb = F.mse_loss(
                        rgb_pred_valid.float(),
                        rgb_target_valid.float(),
                        reduction="mean"
                    )
                    # loss_rgb = F.mse_loss(
                    #     model_pred[:bsz // 2].float(),
                    #     target[:bsz // 2].float(),
                    #     reduction="mean"
                    # )
                    # 后半个batch是BEV数据
                    loss_bev = F.mse_loss(
                        model_pred[bsz // 2:].float(),
                        target[bsz // 2:].float(),
                        reduction="mean"
                    )
                    # 总损失为两者的平均
                    loss = (loss_rgb + loss_bev) / 2.0
                    # loss = loss_bev
                else:
                    # SNR权重计算
                    # snr = compute_snr(noise_scheduler, timesteps)
                    # mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                    #     dim=1
                    # )[0]
                    # if noise_scheduler.config.prediction_type == "epsilon":
                    #     mse_loss_weights = mse_loss_weights / snr
                    # elif noise_scheduler.config.prediction_type == "v_prediction":
                    #     mse_loss_weights = mse_loss_weights / (snr + 1)
                    #
                    # # 分别计算每样本损失
                    # loss_per_sample = F.mse_loss(model_pred.float(), target.float(), reduction='none')
                    # loss_per_sample = loss_per_sample.mean(dim=list(range(1, len(loss_per_sample.shape))))
                    #
                    # # 应用权重
                    # weighted_loss = loss_per_sample * mse_loss_weights
                    # loss_rgb = weighted_loss[:bsz // 2].mean()
                    # loss_bev = weighted_loss[bsz // 2:].mean()
                    # loss = (loss_rgb + loss_bev) / 2.0  # 总损失为平均值
                    # SNR权重计算
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    # 获取batch大小和分割点
                    half = bsz // 2

                    # 分别计算RGB和BEV部分的损失
                    # RGB部分（只使用前4个通道）
                    rgb_pred = model_pred[:half, :4]  # [half, 4, H, W]
                    rgb_target = target[:half, :4]  # [half, 4, H, W]

                    # BEV部分（使用所有通道）
                    bev_pred = model_pred[half:]
                    bev_target = target[half:]

                    # 感知损失的引入

                    # pred_latents_clean_bev = pred_latents_clean[half:]

                    # tensor1, tensor2 = torch.split(pred_latents_clean_bev, split_size_or_sections=4, dim=1)
                    # tensor1 = tensor1.to(dtype=weight_dtype)  # 确保使用正确的数据类型
                    # tensor2 = tensor2.to(dtype=weight_dtype)
              
                    # bev_split1 = vae.decode(tensor1 / vae.config.scaling_factor).sample
                    # bev_split2 = vae.decode(tensor2 / vae.config.scaling_factor).sample
              
                    # reconstructed_bev = torch.cat([bev_split1, bev_split2], dim=1)
                    # reconstructed_bev = (reconstructed_bev / 2 + 0.5).clamp(0, 1)
                    # c = c.to(dtype=weight_dtype)
                    # original_bev = (c / 2 + 0.5).clamp(0, 1)

                    # loss_perceptual = F.binary_cross_entropy(
                    #                                  reconstructed_bev, 
                    #                                  original_bev, 
                    #                                  reduction='mean'
                    # )



                    # 分别计算每样本损失
                    loss_rgb_per_sample = F.mse_loss(rgb_pred.float(), rgb_target.float(), reduction='none')
                    loss_rgb_per_sample = loss_rgb_per_sample.mean(dim=[1, 2, 3])  # 减少到[half]

                    loss_bev_per_sample = F.mse_loss(bev_pred.float(), bev_target.float(), reduction='none')
                    loss_bev_per_sample = loss_bev_per_sample.mean(dim=[1, 2, 3])  # 减少到[half]

                    

                    # BCE分割损失
                    class_weights = torch.tensor([
                        2.,  # drivable_area 
                        7.,  # ped_crossing
                        3.,  # walkway
                        6.,  # carpark_area
                        7.,  # car
                        15.  # truck
                    ], device=seg_outputs.device).float()

                    # 归一化权重
                    class_weights /= class_weights.mean()

                    # 扩展权重张量形状以匹配目标维度：[1, 6, 1, 1] -> [B, 6, H, W] 广播
                    weight_tensor = class_weights.view(1, 6, 1, 1)

                    # 使用带logits的损失函数
                    loss_per_pixel = F.binary_cross_entropy_with_logits(
                        seg_outputs, 
                        seg_target,
                        weight=weight_tensor,  # 应用类别权重
                        reduction='none'       # 保持每个像素的损失
                    )

                    loss_per_pixel_img2bev = F.binary_cross_entropy_with_logits(
                        img2bev_seg_outputs, 
                        seg_target,
                        weight=weight_tensor,  # 应用类别权重
                        reduction='none'       # 保持每个像素的损失
                    )

                    

                    # 计算每个样本的损失（在空间和通道维度平均）
                    loss_per_sample = loss_per_pixel.mean(dim=[1, 2, 3])  # 形状 [B]
                    loss_per_sample_img2bev = loss_per_pixel_img2bev.mean(dim=[1, 2, 3]) 

                    # 确保噪声权重形状正确
                    if sqrt_alpha_bar_t.dim() > 1:
                        sqrt_alpha_bar_t = sqrt_alpha_bar_t.view(-1)  # 移除多余维度

                    if sqrt_alpha_bar_t_img2bev.dim() > 1:
                        sqrt_alpha_bar_t_img2bev = sqrt_alpha_bar_t_img2bev.view(-1)  
                        
                    # 应用噪声感知权重并计算最终损失
                    loss_bce = (sqrt_alpha_bar_t * loss_per_sample).mean() 

                    loss_bce_img2bev = (sqrt_alpha_bar_t_img2bev * loss_per_sample_img2bev).mean() 

                    # loss_bce = (loss_per_sample * mse_loss_weights[half:]).mean() 
                    # 分别应用权重
                    weighted_loss_rgb = loss_rgb_per_sample * mse_loss_weights[:half]
                    weighted_loss_bev = loss_bev_per_sample * mse_loss_weights[half:]

                    # 计算加权损失的平均值
                    loss_rgb = weighted_loss_rgb.mean()
                    loss_bev = weighted_loss_bev.mean()

                    # 总损失为平均值
                    loss = (loss_rgb + loss_bev + loss_bce_img2bev + loss_bce) / 4.0
                    # loss = loss_bev
                # 获取损失值用于后续记录
                loss_rgb_value = loss_rgb.detach().item()
                loss_bev_value = loss_bev.detach().item()
                loss_perceptual_value = loss_bce.detach().item()
                loss_perceptual_value_img2bev = loss_bce_img2bev.detach().item()

                train_loss_rgb += loss_rgb_value
                train_loss_bev += loss_bev_value
                train_loss_perceptual += loss_perceptual_value
                train_loss_perceptual_img2bev += loss_perceptual_value_img2bev

                # 用于全局统计的损失（兼容原有逻辑）
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # 反向传播
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = trainable_params
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 检查是否需要同步梯度
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # ====== 修改点2: 分别记录RGB和BEV的损失 ======
                # 计算梯度累积周期内的平均损失
                # 创建Tensor并移动到加速器设备
                loss_tensor_rgb = torch.tensor(train_loss_rgb / args.gradient_accumulation_steps,
                                               device=accelerator.device)
                loss_tensor_bev = torch.tensor(train_loss_bev / args.gradient_accumulation_steps,
                                               device=accelerator.device)
                loss_tensor_perceptual = torch.tensor(train_loss_perceptual / args.gradient_accumulation_steps,
                                                device=accelerator.device)
                loss_tensor_perceptual_img2bev = torch.tensor(train_loss_perceptual_img2bev / args.gradient_accumulation_steps,
                                                device=accelerator.device)

                # 聚集所有设备上的损失值并计算均值
                avg_loss_rgb = accelerator.gather(loss_tensor_rgb).mean().item()
                avg_loss_bev = accelerator.gather(loss_tensor_bev).mean().item()
                avg_loss_perceptual = accelerator.gather(loss_tensor_perceptual).mean().item()
                avg_loss_perceptual_img2bev = accelerator.gather(loss_tensor_perceptual_img2bev).mean().item()

                # 记录到日志中
                accelerator.log({
                    "train_loss": train_loss,  # 兼容原有总损失
                    "train_loss_rgb": avg_loss_rgb,  # RGB数据损失
                    "train_loss_bev": avg_loss_bev,
                    "train_loss_perceptual": avg_loss_perceptual,
                    "train_loss_perceptual_img2bev": avg_loss_perceptual_img2bev   # BEV数据损失
                }, step=global_step)

                # 重置损失统计
                train_loss = 0.0
                train_loss_rgb = 0.0
                train_loss_bev = 0.0
                train_loss_perceptual = 0.0
                train_loss_perceptual_img2bev = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # 管理检查点数量
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                                
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        
                        # 保存主要检查点（accelerator状态）
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        
                        # 保存LoRA层 - 原有代码保持不变
                        unwrapped_unet = unwrap_model(unet)
                        for lora_name in trainable_loras:
                            cur_save_path = os.path.join(save_path, f"{lora_name}")
                            unet_lora_state_dict = convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(unwrapped_unet, adapter_name=lora_name)
                            )
                            
                            StableDiffusionPipeline.save_lora_weights(
                                save_directory=cur_save_path,
                                unet_lora_layers=unet_lora_state_dict,
                                safe_serialization=True,
                            )
                        
                        # 新增：保存seg_head为pth格式 - 完整实现
                        seg_head_save_path = os.path.join(save_path, "seg_head")
                        os.makedirs(seg_head_save_path, exist_ok=True)
                        
                        # 使用accelerator获取正确的状态字典（支持分布式训练）
                        seg_head_state_dict = accelerator.get_state_dict(seg_head)
                        
                        # 保存为pth格式
                        torch.save(
                            seg_head_state_dict, 
                            os.path.join(seg_head_save_path, "seg_head.pth")
                        )
                        
                        # 可选：记录seg_head的权重文件路径
                        logger.info(f"Saved seg_head weights to {os.path.join(seg_head_save_path, 'seg_head.pth')}")
                        
                        logger.info(f"Saved complete checkpoint to {save_path}")

            # ====== 修改点3: 在进度条中显示详细损失信息 ======
            logs = {
                "step_loss": loss.detach().item(),
                "loss_rgb": loss_rgb_value,  # RGB损失
                "loss_bev": loss_bev_value,
                "loss_perceptual":loss_perceptual_value,
                "loss_perceptual_img2bev":loss_perceptual_value_img2bev,
                "lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        unwrapped_unet = unwrap_model(unet)
        for lora_name in trainable_loras:
            save_dir = os.path.join(args.output_dir, f"{lora_name}")
            unet_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(unwrapped_unet, adapter_name=lora_name)
            )

            StableDiffusionPipeline.save_lora_weights(
                save_directory=save_dir,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )

        state_dict = dict()
        for name, params in unwrapped_unet.named_parameters():
            if params.requires_grad:
                state_dict[name] = params

        save_path = os.path.join(args.output_dir, "model.pth")
        torch.save(state_dict, save_path)

        logger.info(f"Saved final weights to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
