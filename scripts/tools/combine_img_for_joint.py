import os
from PIL import Image, ImageDraw, ImageFont
import argparse

def combine_images(img_folder, gt_folder, pred_folder, output_folder, font_path=None):
    """
    从三个文件夹中读取图像，按文件名匹配后横向拼接保存
    拼接顺序：gt → img → pred
    并在每个图像下方添加标签
    
    参数:
        img_folder: 原始图像文件夹路径
        gt_folder: 真实标注图像文件夹路径
        pred_folder: 预测图像文件夹路径
        output_folder: 拼接图像保存路径
        font_path: 可选字体文件路径
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有图像文件的基本名（不含前缀和后缀）
    base_names = set()
    
    # 处理原始图像文件夹
    for filename in os.listdir(img_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 提取基本名（如"img0" -> "0"）
            base_name = filename.split('imggen_')[-1].split('.')[0]
            base_names.add(base_name)

    try:
        sorted_base_names = sorted([int(name) for name in base_names])
    except ValueError:
        # 如果无法转换为整数，则按字符串排序
        sorted_base_names = sorted(base_names, key=lambda x: int(x) if x.isdigit() else x)

    
    # 尝试加载字体
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, 20)
        else:
            # 尝试加载系统默认字体
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # 处理每个匹配的图像组
    for base_name in sorted_base_names:
        # 构建三个图像文件的完整路径
        target_size = (448, 256) 
        img_path = os.path.join(img_folder, f"imggen_{base_name}.jpg")
        gt_path = os.path.join(gt_folder, f"bev4c_label_{base_name}.png")
        pred_path = os.path.join(pred_folder, f"project4c_label_{base_name}.png")
        
        # 检查所有文件是否存在
        if not all(os.path.exists(p) for p in [img_path, gt_path, pred_path]):
            print(f"跳过不完整的图像组: {base_name}")
            continue
        
        try:
            # 打开所有图像
            img_img = Image.open(img_path)
            gt_img = Image.open(gt_path)
            pred_img = Image.open(pred_path)

            img_img = img_img.resize(target_size, Image.Resampling.LANCZOS)
            pred_img = pred_img.resize(target_size, Image.Resampling.LANCZOS)
            
            # 确保所有图像高度相同（如果不相同则调整）
            heights = [img_img.height, gt_img.height, pred_img.height]
            max_height = max(heights)
            
            # 如果高度不一致，调整所有图像到最大高度
            if len(set(heights)) > 1:
                img_img = img_img.resize((img_img.width, max_height), Image.Resampling.LANCZOS)
                gt_img = gt_img.resize((gt_img.width, max_height), Image.Resampling.LANCZOS)
                pred_img = pred_img.resize((pred_img.width, max_height), Image.Resampling.LANCZOS)
            
            # 添加图像间隔
            spacing = 10  # 间隔宽度（像素）
            
            # 计算拼接后的总宽度（包括间隔）
            total_width = img_img.width + gt_img.width + pred_img.width + 2 * spacing
            
            # 创建新图像 - 增加高度以容纳标签
            label_height = 30  # 标签区域高度
            new_img = Image.new('RGB', (total_width, max_height + label_height), (255, 255, 255))
            
            # 拼接图像 - 修改顺序为 gt → img → pred
            x_offset = 0
            
            # 第一张图像（gt）
            new_img.paste(gt_img, (x_offset, 0))
            draw = ImageDraw.Draw(new_img)
            label_x = x_offset + gt_img.width // 2
            label_y = max_height + 10
            label = "bev_gen"
            draw.text((label_x - len(label)*5, label_y), label, fill="black", font=font)
            x_offset += gt_img.width + spacing
            
            # 第二张图像（img）
            new_img.paste(img_img, (x_offset, 0))
            draw = ImageDraw.Draw(new_img)
            label_x = x_offset + img_img.width // 2
            label_y = max_height + 10
            label = "img_gen"
            draw.text((label_x - len(label)*5, label_y), label, fill="black", font=font)
            x_offset += img_img.width + spacing
            
            # 第三张图像（pred）
            new_img.paste(pred_img, (x_offset, 0))
            draw = ImageDraw.Draw(new_img)
            label_x = x_offset + pred_img.width // 2
            label_y = max_height + 10
            label = "project_bev_to_cam"
            draw.text((label_x - len(label)*5, label_y), label, fill="black", font=font)
            
            # 保存拼接后的图像
            output_path = os.path.join(output_folder, f"combined_{base_name}.png")
            new_img.save(output_path)
            print(f"已保存: {output_path}")
            
        except Exception as e:
            print(f"处理图像组 {base_name} 时出错: {str(e)}")

if __name__ == "__main__":
    combine_images(
        img_folder='/opt/data/private/hwj_autodrive/jointdiff/AAstanderd_imggen_joint/pon_split/imggen_finnal_model_smalltime_25000::29.58/img_gen',
        gt_folder='/opt/data/private/hwj_autodrive/jointdiff/AAstanderd_imggen_joint/pon_split/imggen_finnal_model_smalltime_25000::29.58/bev_with4c',
        pred_folder='/opt/data/private/hwj_autodrive/jointdiff/AAstanderd_imggen_joint/pon_split/imggen_finnal_model_smalltime_25000::29.58/project4c',
        output_folder='/opt/data/private/hwj_autodrive/jointdiff/AAstanderd_imggen_joint/pon_split/imggen_finnal_model_smalltime_25000::29.58/combined',
    )