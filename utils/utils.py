import random
import json
from datetime import datetime
import numpy as np
from PIL import Image, PngImagePlugin

import torch
import torchvision.transforms.v2 as transforms
from peft.tuners.lora.layer import BaseTunerLayer
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from patch import patch

def normalize_image(image):
    return (image - 0.5) / 0.5

def denormalize_image(image):
    return (image + 1) / 2

@torch.no_grad()
def blip2_cap(images, processor = None, model = None, device = "cuda"):

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b") if processor is None else processor
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to("cuda") if model is None else model

    if not isinstance(images[0], Image.Image):
        raw_images = [transforms.ToPILImage()(image) for image in images]
    else:
        raw_images = images

    inputs = processor(raw_images, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=77)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return [text.strip() for text in generated_text]

def save_png_with_comment(image, metadata, output_path):
    # Convert metadata (input variables) to a JSON string
    metadata_str = json.dumps(metadata)
    
    if not isinstance(image, Image.Image):
    # Convert the image array to a PIL image
        image = Image.fromarray(image.astype('uint8'))
    
    # Create a PngInfo object to hold the metadata
    png_info = PngImagePlugin.PngInfo()
    
    # Add the metadata as a comment (tEXt chunk)
    png_info.add_text("Comment", metadata_str)
    
    # Save the image with the metadata comment
    image.save(output_path, "PNG", pnginfo=png_info)

    return output_path

import uuid

def get_combined_filename(name="image", extension="png"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_id = uuid.uuid4().hex[:6]  # Short UUID for brevity
    return f"{name}_{timestamp}_{unique_id}.{extension}"

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_active_adapters(unet):
    for _, module in unet.named_modules():
        if isinstance(module, BaseTunerLayer):
            return module.active_adapter

def set_jointdiff_config_inference(unet, input_pairs, use_cfg = True, batch_size = None, device = "cuda", debug=False):
    """ 设置 jointdiff 模型的全参数推理配置
    
    主要功能：
    1. 配置输入对的联合交叉注意力机制
    2. 准备条件掩码供模型使用
    
    输入格式说明：
    input_pairs = [
        (x索引, y索引, x权重, y权重, 条件名称),
        ...
    ]
    示例：[(0,1,1.0,1.0,"depth")] 表示：
        - 使用第0个输入（RGB）和第1个输入（深度图）进行联合注意力
        - 注意力输出权重均为1.0
        - 使用深度模型处理
    """
    
    # 处理输入配对配置
    attn_config = list(zip(*input_pairs)) 
    # 将数值数据转为Tensor
    for i in range(len(attn_config) - 1):
        attn_config[i] = torch.tensor(attn_config[i])
    x_ids, y_ids, x_weights, y_weights, model_names = attn_config
    # 调整权重张量形状用于广播
    x_weights = x_weights.view(-1, 1, 1)
    y_weights = y_weights.view(-1, 1, 1)

    # 处理CFG扩展（分类器自由引导）
    if use_cfg:
        # 复制并偏移索引：原始输入 + 无条件输入
        x_ids = torch.cat([x_ids, x_ids + batch_size])
        y_ids = torch.cat([y_ids, y_ids + batch_size])
        # 复制权重和模型名
        x_weights, y_weights = torch.cat([x_weights] * 2), torch.cat([y_weights] * 2)
        model_names = model_names * 2
    
    # 准备条件掩码（供模型内部使用）
    cond_masks = {}
    # 为每个独特条件创建布尔掩码
    for cur_cond in set(model_names):
        cond_masks[cur_cond] = [model_name == cur_cond for model_name in model_names]
    
    # 将处理后的配置存入UNet
    attn_config = x_ids.to(device), y_ids.to(device), x_weights.to(device), y_weights.to(device)
    patch.set_jointdiff_config(unet, "attn_config", attn_config)
    patch.set_jointdiff_config(unet, "cond_masks", cond_masks)
    
    # 调试输出
    if debug:
        print("设置注意力配置", attn_config)
        print("设置条件掩码", cond_masks)
    
    return unet

# def set_jointdiff_config_train(unet, input_len, device = "cuda", dtype = torch.float16, debug = False):
#
#     xy_lora = "xy_lora"
#     yx_lora = "yx_lora"
#     xlen = ylen = input_len // 2
#     true_mask = [True] * xlen
#     false_mask = [False] * xlen
#     xy_lora_qo_mask = yx_lora_kv_mask = true_mask + false_mask
#     xy_lora_kv_mask = yx_lora_qo_mask = false_mask + true_mask
#     patch.set_patch_lora_mask(unet, xy_lora, xy_lora_qo_mask, kv_lora_mask = xy_lora_kv_mask)
#     patch.set_patch_lora_mask(unet, yx_lora, yx_lora_qo_mask, kv_lora_mask = yx_lora_kv_mask)
#
#     if debug:
#         print("Set", xy_lora, xy_lora_qo_mask, xy_lora_kv_mask)
#         print("Set", yx_lora, yx_lora_qo_mask, yx_lora_kv_mask)
#
#     y_lora = "y_lora"
#     y_lora_mask = false_mask + true_mask
#     patch.set_patch_lora_mask(unet, y_lora, y_lora_mask)
#     if debug:
#         print("Set", y_lora, y_lora_mask)
#
#     x_ids = torch.arange(xlen).to(device)
#     y_ids = torch.arange(xlen, input_len).to(device)
#     x_weights = torch.ones([1,1,1]).to(device).to(dtype)
#     y_weights = torch.ones([1,1,1]).to(device).to(dtype)
#     attn_config = x_ids, y_ids, x_weights, y_weights
#     patch.set_jointdiff_config(unet, "attn_config", attn_config)
#     if debug:
#         print("Set attn_config", attn_config)
#
#     return unet


def set_jointdiff_config_train(unet, input_len, device="cuda", dtype=torch.float16, debug=False):
    # 定义前半段和后半段长度
    xlen = ylen = input_len // 2
    true_mask = [True] * xlen
    false_mask = [False] * xlen

    # ===========================================================
    # 1. 保持原有 LoRA 命名，但重新解释掩码含义
    # ===========================================================

    # 保持原有 LoRA 名称不变
    xy_lora = "xy_lora"
    yx_lora = "yx_lora"

    # X->Y 方向参数（原 xy_lora）
    xy_lora_qo_mask = true_mask + false_mask  # 仅 X 部分参与 QO
    xy_lora_kv_mask = false_mask + true_mask  # 仅 Y 部分参与 KV

    # Y->X 方向参数（原 yx_lora）
    yx_lora_qo_mask = false_mask + true_mask  # 仅 Y 部分参与 QO
    yx_lora_kv_mask = true_mask + false_mask  # 仅 X 部分参与 KV

    # 使用原有名称设置掩码
    # patch.set_patch_lora_mask(unet, xy_lora, xy_lora_qo_mask, kv_lora_mask=xy_lora_kv_mask)
    # patch.set_patch_lora_mask(unet, yx_lora, yx_lora_qo_mask, kv_lora_mask=yx_lora_kv_mask)

    if debug:
        print(f"{xy_lora} QO Mask (X部分): {xy_lora_qo_mask}")
        print(f"{xy_lora} KV Mask (Y部分): {xy_lora_kv_mask}")
        print(f"{yx_lora} QO Mask (Y部分): {yx_lora_qo_mask}")
        print(f"{yx_lora} KV Mask (X部分): {yx_lora_kv_mask}")

    # ===========================================================
    # 2. 保持 Y-LoRA 设置不变
    # ===========================================================
    y_lora = "y_lora"
    y_lora_mask = false_mask + true_mask
    # patch.set_patch_lora_mask(unet, y_lora, y_lora_mask)
    # if debug:
    #     print(f"{y_lora} Mask: {y_lora_mask}")

    # ===========================================================
    # 3. 索引和权重设置（兼容原有结构）
    # ===========================================================
    x_ids = torch.arange(0, xlen).to(device)
    y_ids = torch.arange(xlen, input_len).to(device)

    # 保持原有权重命名
    x_weights = torch.ones([xlen, 1, 1]).to(device).to(dtype)
    y_weights = torch.ones([ylen, 1, 1]).to(device).to(dtype)

    attn_config = (x_ids, y_ids, x_weights, y_weights)
    patch.set_jointdiff_config(unet, "attn_config", attn_config)

    if debug:
        print("attn_config:")
        print(f"  x_ids: {x_ids.tolist()}")
        print(f"  y_ids: {y_ids.tolist()}")
        print(f"  x_weights: {x_weights.squeeze().tolist()}")
        print(f"  y_weights: {y_weights.squeeze().tolist()}")

    return unet

def parse_schedule(sample_schedule, num_inference_step):
    """ Translate user-friendly schedule to actual sampling schedule.
        Original sample schedule looks like: [seg0,seg1,...] and each seg is [(l_0,r_0),(l_1,r_1),...,(l_n,r_n),with_guidance].
        with_guidance: whether to add guidance for inpainting
        l_i,r_i: an interval in (0,1), will be scaled to (0, N) where N = num_inference_step. It indicates that the i_th input will be denoised from l_i* step to r_i* step in current schedule segment (*: after scaling). 
        A simple example is [[(0,1),(1,1),False]]. Suppose our input is [x,y]. It means sampling x from step 0 to step N (i.e. from pure noise to clean latent) while keeping y at step N (i.e. clean latent), without guidance.
        And the translated schedule will be:
        [
            (0,N-1,0),
            (1,N-1,0),
            ...,
            (N-1,N-1,0)
        ]
    """
    num_inference_step -= 1
    full_schedule = []
    for seg_id, schedule_seg in enumerate(sample_schedule):
        with_guidance = schedule_seg[-1]
        step_segs = []
        step_ranges = []
        for sep_schedule in schedule_seg[:-1]:
            le, ri = int(sep_schedule[0] * num_inference_step), int(sep_schedule[1] * num_inference_step)
            # le, ri = min(num_inference_step - 1, le), min(num_inference_step - 1, ri)
            step_segs.append((le,ri))
            step_ranges.append(ri - le + 1)

        step_num = max(step_ranges)
        for i in range(step_num):
            cur_steps = []
            for step_seg, step_range in zip(step_segs, step_ranges):
                step = step_seg[0] + int(step_range * i / step_num)
                cur_steps.append(step)
            
            if i == 0 and seg_id > 0:
                prev_steps = full_schedule[-1][:-2]
                assert False not in [cur_step == prev_step for cur_step, prev_step in zip(cur_steps, prev_steps)], "Schedule segments should be continuous."
                continue

            full_schedule.append([*cur_steps, with_guidance])
    return full_schedule

def process_images(images, h = None, w = None, verbose = False, div = None, rand_crop = False):
    # Resize and crop image to (h, w) while keeping the aspect ratio

    if isinstance(images[0], Image.Image):
        fh, fw = images[0].height, images[0].width
    else:
        fh, fw = images.shape[-2:]
        assert len(images.shape) >= 3
        if len(images.shape) == 3:
            images = [images]
    
    if h is None and w is None:
        ratio = 1
        h, w = fh, fw
    elif h is None:
        ratio = w / fw
        h = int(fh * ratio)
    elif w is None:
        ratio = h / fh
        w = int(fw * ratio)
    else:
        h_ratio = h / fh
        w_ratio = w / fw
        ratio = max(h_ratio, w_ratio)

    
    if div is not None:
        h = h // div * div
        w = w // div * div
    

    size = (int(fh * ratio + 0.5), int(fw * ratio + 0.5))
    # print(ratio, size)
        # if nw >= w:
        #     size = (h, nw)
        # else:
        #     size = (int(fh / fw * w), w)

    if verbose:
        print(
            f"[INFO] image size {(fh, fw)} resize to {size} and centercrop to {(h, w)}")

    image_ls = []
    for image in images:
        if ratio <= 1 and rand_crop:
            resized_frame = image
        else:
            resized_frame = transforms.Resize(size, antialias=True)(image)
        if rand_crop:
            cropped_frame = transforms.RandomCrop([h, w])(resized_frame)
        else:
            cropped_frame = transforms.CenterCrop([h, w])(resized_frame)
        
        image_ls.append(cropped_frame)
    if isinstance(images[0], Image.Image):
        return image_ls
    else:
        return torch.stack(image_ls)


# ToTensor = transforms.Compose(
#                 [
#                     transforms.ToImage(),
#                     transforms.ToDtype(torch.float32, scale=True),
#                 ]
#             )

# ToPILImage = transforms.ToPILImage()

def get_prompt(masks, car_counts=None, truck_counts=None):
        """
        检查批量的多通道二值掩码中各通道是否全零，返回格式化的描述字符串
        
        参数:
        masks: 输入的批量多通道二值掩码，形状为：
            [批次数, 通道数, 高度, 宽度] 或 
            [批次数, 高度, 宽度, 通道数] 或 
            [通道数, 高度, 宽度] (单样本)
            通道顺序：drivable_area, ped_crossing, walkway, carpark, car, truck
        
        car_counts: 可选的汽车数量列表，每个元素对应一个批次（可以是张量或列表）
        truck_counts: 可选的卡车数量列表，每个元素对应一个批次（可以是张量或列表）
        
        返回:
        (x_strings, y_strings) 元组，每个元素为字符串列表，格式分别为：
            x_string: "a realistic driving scene with drivable_area，ped_crossing, three cars and two trucks"
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
        
        # 数字到英文单词的映射字典
        number_words = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
            11: "eleven",
            12: "twelve",
            13: "thirteen",
            14: "fourteen",
            15: "fifteen",
            16: "sixteen",
            17: "seventeen",
            18: "eighteen",
            19: "nineteen",
            20: "twenty"
        }
        
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
        
        # 获取批量大小
        batch_size = masks.shape[0]
        
        # 初始化描述字符串列表
        x_strings = []
        y_strings = []
        
        # 处理车辆计数输入（如果是张量则转换为列表）
        if car_counts is not None and isinstance(car_counts, torch.Tensor):
            car_counts = car_counts.cpu().numpy().tolist()
        if truck_counts is not None and isinstance(truck_counts, torch.Tensor):
            truck_counts = truck_counts.cpu().numpy().tolist()
        
        # 处理每个样本
        for i in range(batch_size):
            sample_mask = masks[i]
            nonzero_labels = []
            vehicle_counts = []  # 用于汽车和卡车的数量信息
            
            # 检查每个通道是否非零
            for c in range(6):
                channel = sample_mask[c]
                if not torch.all(channel == 0):
                    label = channel_labels[c]
                    nonzero_labels.append(label)
                    
                    # 如果是车辆类别且提供了数量信息
                    if label in ["car", "truck"]:
                        # 获取对应数量
                        if label == "car" and car_counts is not None and i < len(car_counts):
                            count = car_counts[i]
                            # 确保count是整数
                            if isinstance(count, torch.Tensor):
                                count = count.item()
                            # 将数字转换为英文单词
                            if count in number_words:
                                count_word = number_words[count]
                            else:
                                count_word = str(count)  # 超出范围保持数字
                            vehicle_counts.append(f"{count_word} cars")
                        elif label == "truck" and truck_counts is not None and i < len(truck_counts):
                            count = truck_counts[i]
                            # 确保count是整数
                            if isinstance(count, torch.Tensor):
                                count = count.item()
                            # 将数字转换为英文单词
                            if count in number_words:
                                count_word = number_words[count]
                            else:
                                count_word = str(count)  # 超出范围保持数字
                            vehicle_counts.append(f"{count_word} trucks")
            
            # 为当前样本构建描述字符串
            if nonzero_labels:
                labels_str = "，".join(nonzero_labels)
                
                # 添加车辆数量描述
                if vehicle_counts:
                    vehicles_str = " and ".join(vehicle_counts)
                    x_string = f"a realistic driving scene with {labels_str}，including {vehicles_str}"
                    y_string = f"a seg mask with {labels_str}，including {vehicles_str}"
                else:
                    x_string = f"a realistic driving scene with {labels_str}"
                    y_string = f"a seg mask with {labels_str}"
                
            else:
                x_string = "a realistic driving scene"
                y_string = "a seg mask"
            
            x_strings.append(x_string)
            y_strings.append(y_string)
        
        return x_strings, y_strings