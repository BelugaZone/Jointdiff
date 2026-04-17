from pathlib import Path
import sys
_JOINTDIFF_ROOT = Path(__file__).resolve().parents[2]
if str(_JOINTDIFF_ROOT) not in sys.path:
    sys.path.insert(0, str(_JOINTDIFF_ROOT))

import json
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from multi_view_generation.bev_utils.nuscenes_dataset import NuScenesDataset

class NuSceneCaptioning:
    def __init__(self, processor_path, model_path, device='cuda'):
        self.processor = BlipProcessor.from_pretrained(processor_path)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path).to(device)
        self.device = device
        
    def __call__(self, image):
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)
    
    def batch_caption(self, images):
        """批量处理图像生成描述"""
        inputs = self.processor(images, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.generate(**inputs)
        return self.processor.batch_decode(outputs, skip_special_tokens=True)

def prepare_image(img):
    """将各种格式的图像转换为PIL格式"""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().squeeze()
        
        # 处理不同的张量形状
        if img.ndim == 4:  # 批处理维度
            img = img[0]
        if img.ndim == 3 and img.shape[0] == 3:  # CHW格式
            img = img.permute(1, 2, 0)
        
        # 值域转换
        if img.max() <= 1.0:
            img = img * 255
        img = img.numpy().astype(np.uint8)
        
    if isinstance(img, np.ndarray):
        # 转换numpy数组到PIL Image
        if img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)
        if img.ndim == 2:  # 灰度图
            img = np.stack([img]*3, axis=-1)
        elif img.shape[0] == 3:  # C×H×W转H×W×C
            img = img.transpose(1, 2, 0)
    
    return Image.fromarray(img)

def generate_and_save_captions(dataset, output_path, batch_size=16):
    # 初始化描述生成模型
    captioner = NuSceneCaptioning(
        processor_path="Salesforce/blip-image-captioning-base",
        model_path="Salesforce/blip-image-captioning-base"
    )
    
    # 创建数据加载器 - 使用单进程避免多线程问题
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # 单进程避免潜在冲突
    )
    
    results = []
    sample_count = 0
    
    # 使用DataLoader迭代处理
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating Captions")):
        # 准备图像
        pil_images = []
        for i in range(len(batch["image"])):
            img_tensor = batch["image"][i]
            if isinstance(img_tensor, torch.Tensor):
                # 确保图像格式正确
                img_tensor = img_tensor.permute(2, 0, 1) if img_tensor.shape[-1] == 3 else img_tensor
            pil_images.append(prepare_image(img_tensor))
        
        # 批量生成描述
        try:
            captions = captioner.batch_caption(pil_images)
        except Exception as e:
            print(f"Error generating batch captions for batch {batch_idx}: {e}")
            # 回退单张处理
            captions = []
            for img in pil_images:
                try:
                    captions.append(captioner(img))
                except:
                    captions.append("Error generating caption")
        
        # 收集结果
        for i in range(len(batch["image"])):
            token = batch.get("token", [str(sample_count)])[i] if isinstance(batch.get("token"), list) else batch.get("token", str(sample_count))
            scene_name = batch.get("scene_name", ["unknown"])[i] if isinstance(batch.get("scene_name"), list) else batch.get("scene_name", "unknown")
            timestamp = batch.get("timestamp", [0])[i] if isinstance(batch.get("timestamp"), list) else batch.get("timestamp", 0)
            
            caption = captions[i]
            print(f"Sample {sample_count}: {caption}")
            
            results.append({
                "sample_idx": sample_count,
                "token": token,
                "caption": caption,
                "scene_name": scene_name,
                "timestamp": timestamp
            })
            sample_count += 1
    
    # 保存结果到JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印摘要信息
    success_count = sum(1 for r in results if not r['caption'].startswith("Error"))
    print(f"\nSummary:")
    print(f" - Total samples processed: {len(results)}")
    print(f" - Successful captions generated: {success_count}")
    print(f" - Failed captions: {len(results) - success_count}")
    print(f" - Results saved to: {output_path}")
    
    return results

# 使用示例
if __name__ == "__main__":
    # 确保您已经正确实现了NuScenesDataset类
    
    # 尝试加载分割图像以避免KeyError
    train_dataset = NuScenesDataset(
        split=0,
        return_cam_img=True,
        return_bev_img=True,
        return_seg_img=False,  # 设置True以避免'segmentation'键错误
        return_all_cams=False,
        cam_res=(512, 512),
        dataset_dir="/opt/data/private/nuscenes"
    )
    
    print("Starting caption generation for NuScenes dataset...")
    captions = generate_and_save_captions(
        dataset=train_dataset,
        output_path="nuscenes_captions.json",
        batch_size=4  # 较小的批次大小更稳定
    )
    
    # 打印前5个生成的描述作为示例
    print("\nExample captions:")
    for caption_info in captions[:5]:
        print(f"Sample {caption_info['sample_idx']}: {caption_info['caption']}")
    
    print(f"\nCompleted! Generated {len(captions)} captions saved to nuscenes_captions.json")