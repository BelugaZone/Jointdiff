import argparse
import torch.nn as nn
import logging
import math
import os
import random
import shutil
from pathlib import Path
import pdb

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import Image
from packaging import version
from peft import LoraConfig
import torchvision.transforms.v2 as transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, PNDMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from patch import patch
from utils.utils import set_unicon_config_train
from utils.load_utils import load_image_folder
from utils.peft_utils import get_peft_model_state_dict, set_adapters_requires_grad
from safetensors import safe_open
from multi_view_generation.bev_utils.cityscapes_dataset import CityscapesDataset
from multi_view_generation.bev_utils.nuscenes_dataset import NuScenesDataset
from multi_view_generation.bev_utils import viz_bev
from multi_view_generation.bev_utils import bev_pixels_to_cam , render_map_in_image, cam_pixels_to_bev
from peft import LoraConfig, get_peft_model

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="UniCon training script.")
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
        default="unicon",
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
        default=10,
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
    # UniCon specific arguments
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


def main():

    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=args.joint_dropout_prob > 0)
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






    print("加载PNDM调度器...")
    noise_scheduler = PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    print("加载tokenizer ...")
    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="tokenizer", revision=None
    )
    print("加载 text_encoder...")
    text_encoder = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="text_encoder", revision=None
    )

    print("加载原始vae...")
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae", revision=None, variant=None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet_class = UNet2DConditionModel

    unet = unet_class.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet", revision=None, variant=None
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

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

    def modify_unet_for_8_channels(unet: UNet2DConditionModel):
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

    unet = modify_unet_for_8_channels(unet)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # all_loras = []

    if args.train_y_lora:
        y_lora_config = LoraConfig(
            r=args.rank if args.ylora_rank is None else args.ylora_rank,
            lora_alpha=args.rank if args.ylora_rank is None else args.ylora_rank,
            init_lora_weights="gaussian",
            target_modules=[
                "attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0",
                "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0",
            ],
        )
        # unet = get_peft_model(unet, y_lora_config)

    # patch.hack_lora_forward(unet)
    #
    # unet.set_adapters(all_loras)
    #
    # unet = get_peft_model(unet, lora_config)

    # param_groups = classify_unet_params(unet)
    # for param in param_groups["frozen"]:
    #     param.requires_grad = False

    # trainable_loras = all_loras
    # set_adapters_requires_grad(unet, True, trainable_loras)
    

    new_layer_params = []
    for name, param in unet.named_parameters():
        # if "conv_in" in name or "conv_out" in name or "lora" in name:
            new_layer_params.append(param)
            # print(name)
            param.requires_grad = True  # 强制设置为可训练

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
        if unet is None:
            logger.error("未找到unet，跳过")
            return

        for name, params in unet.named_parameters():
            if params.requires_grad:
                state_dict[name] = params

        save_path = os.path.join(output_dir, "model.pth")
        torch.save(state_dict, save_path)

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
        y_tensor = ((y_img_pil + 1) / 2).clip(0, 1).permute(1, 2, 0).cpu().numpy()
        y_tensor = (y_tensor > 0.5)
        channel_coords = bev_pixels_to_cam(y_tensor, E01)
        bev_cam = render_map_in_image(intrinsics01, channel_coords)
        bev_cam_imge = viz_bev(bev_cam)
        colored_seg = viz_bev(y_tensor)  # 假设使用 nuScenes 数据集
        return colored_seg.pil, bev_cam_imge.pil

    accelerator.register_load_state_pre_hook(load_model_hook)
    accelerator.register_save_state_pre_hook(save_model_hook)

    trainable_params = filter(lambda p: p.requires_grad, unet.parameters())

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
        cam_res=256,
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
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
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
        accelerator.init_trackers("unicon", config={k: v if not isinstance(v, list) else " ".join(v) for k, v in
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

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    logger.info(f"VAE scale factor: {vae_scale_factor}")

    bsz = 0

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        unet = unet.to(torch.float32)

        # 初始化损失统计
        train_loss = 0.0
        train_loss_rgb = 0.0  # 新增：RGB数据损失统计
        train_loss_bev = 0.0  # 新增：BEV数据损失统计
        logger.info(f"Epoch {epoch}, Step {global_step}")

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                if args.rand_transform:
                    train_transforms = random_train_transforms()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                batch["image"] = batch["image"].permute(0, 3, 1, 2).to(device)  # 图像
                batch["segmentation"] = batch["segmentation"].permute(0, 3, 1, 2).to(device)  # 分割图
                batch_zero = batch["segmentation"].shape[0]

                if bsz != batch["segmentation"].shape[0]:
                    bsz = batch["segmentation"].shape[0]

                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
                timesteps = timesteps.long()

                c = batch["segmentation"]
                c = c[:, :6, :, :]
                # # 处理BEV数据
                # tensor1, tensor2, tensor3, tensor4 = torch.split(c, split_size_or_sections=3, dim=1)
                tensor1, tensor2 = torch.split(c, split_size_or_sections=3, dim=1)
                c1 = vae.encode(tensor1.to(dtype=weight_dtype)).latent_dist.sample()
                c2 = vae.encode(tensor2.to(dtype=weight_dtype)).latent_dist.sample()
                # c3 = vae.encode(tensor3.to(dtype=weight_dtype)).latent_dist.sample()
                # c4 = vae.encode(tensor4.to(dtype=weight_dtype)).latent_dist.sample()
                # latents_bev = torch.cat([c1, c2, c3, c4], dim=1)
                latents_bev = torch.cat([c1, c2], dim=1)

                # latents_bev = vae.encode(c.to(dtype=weight_dtype)).latent_dist.sample()

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
                # 拼接RGB和BEV数据
                # latents = torch.cat([latents_16ch, latents_bev], dim=0)
                latents = latents_rgb
                # latents = latents_16ch

                # #####应用掩码
                
                original_bsz = latents_rgb.size(0)
                rgb_mask = torch.zeros(1, 8, 1, 1, device=latents_rgb.device, dtype=latents_rgb.dtype)
                rgb_mask[:, :4, :, :] = 1.0  # RGB有效通道（前4个）
                bev_mask = torch.ones(1, 8, 1, 1, device=latents_rgb.device, dtype=latents_rgb.dtype)  # BEV全部通道有效
                
                # 创建整个批次的掩码
                # mask = torch.cat([rgb_mask] * original_bsz + [bev_mask] * original_bsz, dim=0)
                mask = torch.cat([rgb_mask] * original_bsz,dim=0)

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

                # 条件嵌入（文本）
                if args.prompt_dropout_prob > 0:
                    encoder_hidden_states = torch.randn(batch_zero, 77, 768)
                    null_conditioning = torch.zeros_like(encoder_hidden_states)
                    encoder_hidden_states = null_conditioning.to(device)


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

                # # 应用掩码
                masked_input = ChannelMaskGrad.apply(input_data, mask)

                # 预测噪声残差
                model_pred = unet(masked_input, timesteps, encoder_hidden_states, return_dict=False)[0]

                # ====== 修改点1: 分别计算RGB和BEV的损失 ======
                if args.snr_gamma is None:
                    loss_bev = F.mse_loss(
                        model_pred[:,:4,:,:].float(),
                        target[:,:4,:,:].float(),
                        reduction="mean"
                    )

                else:
                    # SNR权重计算
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    # 分别计算每样本损失
                    loss_per_sample = F.mse_loss(model_pred[:,:4,:,:].float(), target[:,:4,:,:].float(), reduction='none')
                    loss_per_sample = loss_per_sample.mean(dim=list(range(1, len(loss_per_sample.shape))))


                    # 应用权重
                    weighted_loss = loss_per_sample * mse_loss_weights
                    loss_bev = weighted_loss.mean()
                    loss = loss_bev

                # 获取损失值用于后续记录
                loss_bev_value = loss_bev.detach().item()
                train_loss_bev += loss_bev_value

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

                loss_tensor_bev = torch.tensor(train_loss_bev / args.gradient_accumulation_steps,
                                               device=accelerator.device)

                avg_loss_bev = accelerator.gather(loss_tensor_bev).mean().item()

                # 记录到日志中
                accelerator.log({
                    "train_loss": train_loss,  # 兼容原有总损失
                    "train_loss_bev": avg_loss_bev  # BEV数据损失
                }, step=global_step)

                # 重置损失统计
                train_loss = 0.0
                train_loss_bev = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
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

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        # 保存LoRA层
                        # unwrapped_unet = unwrap_model(unet)
                        # for lora_name in trainable_loras:
                        #     cur_save_path = os.path.join(save_path, f"{lora_name}")
                        #     unet_lora_state_dict = convert_state_dict_to_diffusers(
                        #         get_peft_model_state_dict(unwrapped_unet, adapter_name=lora_name)
                        #     )
                        #
                        #     StableDiffusionPipeline.save_lora_weights(
                        #         save_directory=cur_save_path,
                        #         unet_lora_layers=unet_lora_state_dict,
                        #         safe_serialization=True,
                        #     )
                        # === 修改点1：更新LoRA权重保存方法 ===
                        # 不再需要遍历多个适配器，因为只有一个适配器
                        unwrapped_unet = accelerator.unwrap_model(unet)

                        # 获取LoRA权重（使用默认适配器名称）
                        lora_state_dict = get_peft_model_state_dict(unwrapped_unet)

                        # 转换为Diffusers格式
                        unet_lora_state_dict = convert_state_dict_to_diffusers(lora_state_dict)

                        # 保存LoRA权重
                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=os.path.join(save_path, "lora"),
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )

                        logger.info(f"Saved checkpoint to {save_path}")

            # ====== 修改点3: 在进度条中显示详细损失信息 ======
            logs = {
                "step_loss": loss.detach().item(),
                "loss_bev": loss_bev_value,  # BEV损失
                "lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break



    # Save the lora layers
    # accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     unet = unet.to(torch.float32)
    #
    #     unwrapped_unet = unwrap_model(unet)
    #     for lora_name in trainable_loras:
    #         save_dir = os.path.join(args.output_dir, f"{lora_name}")
    #         unet_lora_state_dict = convert_state_dict_to_diffusers(
    #             get_peft_model_state_dict(unwrapped_unet, adapter_name=lora_name)
    #         )
    #
    #         StableDiffusionPipeline.save_lora_weights(
    #             save_directory=save_dir,
    #             unet_lora_layers=unet_lora_state_dict,
    #             safe_serialization=True,
    #         )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

            # 获取完整模型
        unwrapped_unet = accelerator.unwrap_model(unet)

            # 保存LoRA权重
        lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(lora_state_dict)

            # 创建保存目录
        lora_save_dir = os.path.join(args.output_dir, "lora")
        os.makedirs(lora_save_dir, exist_ok=True)

            # 保存LoRA权重
        StableDiffusionPipeline.save_lora_weights(
                save_directory=lora_save_dir,
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