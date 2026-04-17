"""
在「与 train_jointdiff_new_full_withcamparam_doubleloss_imgae2bev.py 相同依赖」的环境下运行，
打印 UNet、seg_head 及合计的 requires_grad=True 参数量。

用法（在 Jointdiff 仓库根目录下）:
  python scripts/tools/count_trainable_params_doubleloss.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5
"""
from __future__ import annotations

from pathlib import Path
import sys
_JOINTDIFF_ROOT = Path(__file__).resolve().parents[2]
if str(_JOINTDIFF_ROOT) not in sys.path:
    sys.path.insert(0, str(_JOINTDIFF_ROOT))

import argparse

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

from patch import patch
from seghead.segdecoder_double_loss import seg_decorder


def classify_unet_params(unet: nn.Module):
    params_dict = {
        "new_input_layer": [],
        "new_output_layer": [],
        "frozen": [],
    }
    new_input_names = []
    if unet.conv_in.weight.shape[1] > 4:
        new_input_names.append("conv_in.weight")
        if hasattr(unet.conv_in, "bias") and unet.conv_in.bias is not None:
            new_input_names.append("conv_in.bias")

    new_output_names = []
    if unet.conv_out.weight.shape[0] > 4:
        new_output_names.append("conv_out.weight")
        if hasattr(unet.conv_out, "bias") and unet.conv_out.bias is not None:
            new_output_names.append("conv_out.bias")

    for name, param in unet.named_parameters():
        if name in new_input_names:
            params_dict["new_input_layer"].append(param)
        elif name in new_output_names:
            params_dict["new_output_layer"].append(param)
        else:
            params_dict["frozen"].append(param)
    return params_dict


def modify_unet_for_8_channels(unet: UNet2DConditionModel) -> UNet2DConditionModel:
    original_conv_in_weight = unet.conv_in.weight.data
    original_conv_in_bias = unet.conv_in.bias.data
    new_conv_in = nn.Conv2d(
        8,
        original_conv_in_weight.shape[0],
        kernel_size=3,
        padding=1,
    )
    with torch.no_grad():
        new_conv_in.weight[:, :4] = original_conv_in_weight
        new_conv_in.weight[:, 4:] = original_conv_in_weight
        new_conv_in.bias.data = original_conv_in_bias
    unet.conv_in = new_conv_in

    original_conv_out_weight = unet.conv_out.weight.data
    original_conv_out_bias = unet.conv_out.bias.data
    new_conv_out = nn.Conv2d(
        original_conv_out_weight.shape[1],
        8,
        kernel_size=3,
        padding=1,
    )
    with torch.no_grad():
        new_conv_out.weight[:4] = original_conv_out_weight
        new_conv_out.weight[4:] = original_conv_out_weight
        new_conv_out.bias.data[:4] = original_conv_out_bias
        new_conv_out.bias.data[4:] = original_conv_out_bias
    unet.conv_out = new_conv_out
    unet.config.in_channels = 8
    unet.config.out_channels = 8
    return unet


def freeze_specific_parameters(model: nn.Module) -> None:
    frozen_names = (
        "conv_out.bias",
        "conv_out.weight",
        "conv_norm_out.bias",
        "conv_norm_out.weight",
    )
    for name, param in model.named_parameters():
        if name in frozen_names:
            param.requires_grad = False


def count_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="与训练脚本相同的 UNet 预训练路径",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    args = parser.parse_args()

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )
    unet = modify_unet_for_8_channels(unet)

    for param in classify_unet_params(unet)["frozen"]:
        param.requires_grad = False

    patch.apply_patch(unet, name_skip=None, train=True)
    patch.initialize_joint_layers(unet, post="conv")

    for param in classify_unet_params(unet)["frozen"]:
        param.requires_grad = False

    patch.set_joint_layer_requires_grad(unet, ["xy_lora", "yx_lora"], True)

    for _, param in unet.named_parameters():
        param.requires_grad = True

    freeze_specific_parameters(unet)

    seg_head = seg_decorder((256, 256))

    ut = count_trainable(unet)
    st = count_trainable(seg_head)
    tot = ut + st
    all_u = sum(p.numel() for p in unet.parameters())
    all_s = sum(p.numel() for p in seg_head.parameters())

    print("=== 与训练脚本 optimizer 一致：requires_grad=True 参数量 ===")
    print(f"UNet 可训练:     {ut:,}")
    print(f"seg_head 可训练: {st:,}")
    print(f"合计可训练:      {tot:,}")
    print("---")
    print(f"UNet 总参数:     {all_u:,}")
    print(f"seg_head 总参数: {all_s:,}")
    print(f"UNet 可训练占比: {100.0 * ut / all_u:.4f}%")


if __name__ == "__main__":
    main()
