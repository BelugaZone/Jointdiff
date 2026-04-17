from pathlib import Path
import sys
_JOINTDIFF_ROOT = Path(__file__).resolve().parents[2]
if str(_JOINTDIFF_ROOT) not in sys.path:
    sys.path.insert(0, str(_JOINTDIFF_ROOT))

import torch
import numpy as np
import os
from tqdm import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel
from multi_view_generation.bev_utils.nuscenes_dataset import NuScenesDataset  # 假设您的数据集类在此模块中

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# 配置参数
PRETRAINED_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
CONTROLNET_PATH = "/opt/data/private/hwj_autodrive/jointdiff/controlnet_training_results_3c/checkpoints/epoch_41"    # 训练好的ControlNet路径
OUTPUT_DIR = "/opt/data/private/hwj_autodrive/jointdiff/controlnet_training_results_3c/checkpoints/epoch_41"  # 输出目录
CONDITION_DIR = os.path.join(OUTPUT_DIR, "conditions")  # 条件图像目录
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")  # 结果图像目录
BATCH_SIZE = 4  # 批量生成大小
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.5
SEED = 42

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONDITION_DIR, exist_ok=True)  # 创建条件图像目录
os.makedirs(RESULT_DIR, exist_ok=True)  # 创建结果图像目录

# 加载验证数据集
val_dataset = NuScenesDataset(
    split=1,  # 假设1是验证集
    return_cam_img=True,
    return_bev_img=True,
    return_seg_img=True,
    augment_cam_img=False,
    augment_bev_img=False,
    return_all_cams=False,
    mini_dataset=False,
    metadrive_compatible=False,
    cam_res=(256, 256),
    dataset_dir="/opt/data/private/nuscenes"
)

# 创建数据加载器
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # 保持原始顺序
    num_workers=4,
    pin_memory=True
)

# 加载预训练模型组件
tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_MODEL_NAME, subfolder="text_encoder").to(device)

# 加载训练好的ControlNet (6通道输入)
controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_PATH,
    # conditioning_channels=6  # 关键：指定6通道输入
).to(device)

# 创建ControlNet管道
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    PRETRAINED_MODEL_NAME,
    controlnet=controlnet,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    safety_checker=None,  # 禁用安全检查器以加快速度
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

target_dtype = torch.float16 if device == "cuda" else torch.float32

# 转换ControlNet
controlnet = controlnet.to(target_dtype)

# 转换UNet
pipe.unet = pipe.unet.to(target_dtype)

# 转换文本编码器
text_encoder = text_encoder.to(target_dtype)

# 确保管道使用正确的精度
pipe = pipe.to(target_dtype)

# 验证精度统一
print(f"ControlNet dtype: {next(controlnet.parameters()).dtype}")
print(f"UNet dtype: {next(pipe.unet.parameters()).dtype}")
print(f"Text Encoder dtype: {next(text_encoder.parameters()).dtype}")

# 启用优化
if device == "cuda":
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

# 固定随机种子
generator = torch.Generator(device).manual_seed(SEED)

# 批量生成函数
def generate_batch(seg_maps):
    """生成一批图像"""
    # 将分割图转换为张量，并匹配模型精度
    seg_tensors = torch.from_numpy(seg_maps).permute(0, 3, 1, 2).to(device)
    
    # 获取模型的精度
    model_dtype = next(pipe.unet.parameters()).dtype
    
    # 将输入数据转换为模型精度
    seg_tensors = seg_tensors.to(model_dtype)
    
    # 生成图像
    images = pipe(
        prompt=[""] * len(seg_maps),
        image=seg_tensors,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator
    ).images
    
    return images

# 处理整个验证集
print(f"开始生成验证集图像，共 {len(val_dataset)} 个样本...")

for batch_idx, batch in enumerate(tqdm(val_loader, desc="处理验证集")):
    seg_maps = batch["segmentation"].numpy()[:, :, :, :3]  # [B, H, W, 6]
    
    # 转换为模型精度
    model_dtype = next(pipe.unet.parameters()).dtype
    seg_maps = seg_maps.astype(np.float16 if model_dtype == torch.float16 else np.float32)
    
    # 生成图像
    generated_images = generate_batch(seg_maps)
    
    # 保存结果
    for i, img in enumerate(generated_images):
        # 计算全局索引
        sample_idx = batch_idx * BATCH_SIZE + i
        
        # 获取原始图像路径作为参考
        img_path = batch["image_path"][i] if "image_path" in batch else f"sample_{sample_idx}"
        
        # 创建文件名
        filename_base = Path(img_path).stem
        
        # 保存生成的图像到结果目录
        result_filename = f"{filename_base}_generated.png"
        img.save(os.path.join(RESULT_DIR, result_filename))
        
        # 保存条件输入（分割图）到条件目录
        condition_filename = f"{filename_base}_condition.png"
        seg_img = (seg_maps[i] * 255).astype(np.uint8)[:, :, :3]  # 取前3通道可视化
        Image.fromarray(seg_img).save(os.path.join(CONDITION_DIR, condition_filename))

print(f"所有验证图像已保存到以下目录:")
print(f"- 条件图像: {CONDITION_DIR}")
print(f"- 生成结果: {RESULT_DIR}")