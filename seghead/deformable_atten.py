from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from patch.deformable_detr.position_encoding import LearnedPositionalEncoding
from mmengine.model import xavier_init, constant_init
# from mmengine.model.wrappers.utils import force_fp32
from mmengine.model import BaseModule, ModuleList, Sequential  # 基础模块
import cv2
import os
import time

# from mmengine.utils import ext_loader 
# from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
#     MultiScaleDeformableAttnFunction_fp16
# from projects.mmdet3d_plugin.models.utils.bricks import run_time
# ext_module = ext_loader.load_ext(
#     '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])






class SpatialCrossAttention(BaseModule):
    """Optimized version for single-camera scenarios"""

    def __init__(self,
                 embed_dims=256,
                 num_cams=1,  # 强制设置为1，无需修改
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 feat_size_h=None,
                 feat_size_w=None,
                 img_size=None,
                 **kwargs
                 ):
        # 保持原有初始化，但注意num_cams已被设置为1
        super().__init__(init_cfg)
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = DeformableAttention3D(embed_dims=256)
        self.embed_dims = embed_dims
        self.num_cams = 1  # 确保单摄像头处理
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()
        self.feat_size_h = feat_size_h
        self.feat_size_w = feat_size_w
        self.position_embedding = LearnedPositionalEncoding(128, feat_size_h, feat_size_w)   # dim = embed_dims//2
        self.img_size = img_size

    def init_weight(self):
        xavier_init(self.output_proj, distribution='uniform', bias=0.)



    def get_reference_points(self, H, W, Z=8, num_points_in_pillar=4,bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        
        zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                            device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1) # [4, 150, 150, 3]
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1) # [4,22500,3]
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)  # [1, 4, 22500, 3]
        return ref_3d #

        


   


    def point_sampling(self, reference_points, pc_range, cam2img, img_shape):
    
        # 1. 从图像元数据中提取单个摄像头的lidar2img变换矩阵
        # cam2img = []
        # for img_meta in img_metas:
        #     # 单摄像头场景：只取第一个摄像头的变换矩阵
        #     lidar2img.append(img_meta['lidar2img'][0])  # 只取第一个摄像头
        # cam2img = np.asarray(cam2img)
        
        # # 2. 将变换矩阵转换为张量
        # cam2img = reference_points.new_tensor(cam2img)  # (B, 4, 4)
        # 现在形状为 [batch_size, 4, 4]
        
        # 3. 克隆参考点以避免修改原始数据
        reference_points = reference_points.clone()  # [1, 4, 22500, 3]
        
        # 4. 将归一化坐标映射到真实世界坐标
        # X坐标: 从[0,1]映射到[pc_range[0], pc_range[3]]
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        
        # Y坐标: 从[0,1]映射到[pc_range[1], pc_range[4]]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        
        # Z坐标: 从[0,1]映射到[pc_range[2], pc_range[5]]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]



        # print('reference_points', reference_points.shape)
        # reference_points = reference_points.repeat(b, 1, 1, 1)

        # 5. 添加齐次坐标分量（w=1）
        # reference_points = torch.cat(
        #     (reference_points, torch.ones_like(reference_points[..., :1])), -1) 
        # 现在形状为 [1, 4, 22500, 4]
        
        # 6. 调整维度顺序：从 [B, D, num_query, 4] 变为 [D, B, num_query, 4]
        # reference_points = reference_points.permute(1, 0, 2, 3)
        # 现在形状为 [4, 1, 22500, 4]
        
        # 7. 获取维度信息
        # D, B, num_query = reference_points.size()[:3]  # D=4, B=1, num_query=22500
        
        # 8. 单摄像头优化：不需要扩展摄像头维度
        # 直接添加最后一个维度用于矩阵乘法
        # reference_points = reference_points.unsqueeze(-1)  # [4, 1, 22500, 4, 1]
        
        # 9. 扩展变换矩阵以匹配参考点形状
        # cam2img = cam2img.view(1, B, 1, 4, 4).repeat(D, 1, num_query, 1, 1)
        # 现在形状为 [4, 1, 22500, 4, 4]
        
        # 10. 执行投影变换：将3D点投影到2D图像平面
       

        f, cu, cv = cam2img[:, 0, 0], cam2img[:, 0, 2], cam2img[:, 1, 2]
        x, z, y = reference_points[..., 0], reference_points[..., 1], reference_points[..., 2]
        f = f[:, None, None]
        cu = cu[:, None, None]
        cv = cv[:, None, None]
        z = torch.clamp(z, min=0.1)
        v = (y * f) / z + cv  # [b, num_patch] height
        u = (x * f) / z + cu  # [b, num_patch] width
        reference_points_cam = torch.stack([u, v], dim=-1) # [b, num_patch, 2]
        # print("reference_points_cam after cam2img",reference_points_cam.shape)

        # reference_points_cam = torch.matmul(cam2img.to(torch.float32),
        #                                     reference_points.to(torch.float32)).squeeze(-1)
        # 结果形状为 [4, 1, 22500, 4]
        # 最后一维是齐次坐标 (x, y, z, w)
        
        # 11. 设置小量防止除零
        # eps = 1e-5
        
        # 12. 创建深度掩码（z > 0 表示点在相机前方）
        # bev_mask = (reference_points_cam[..., 2:3] > eps)
        
        # 13. 将齐次坐标转换为笛卡尔坐标
        # reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        #     reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
        # 现在形状为 [4, 1, 22500, 2]
        
        # 14. 单摄像头优化：直接使用第一个摄像头的图像尺寸
        
        # 归一化到[0,1]范围（除以图像尺寸）
        reference_points_cam[..., 0] /= img_shape[1]  # 宽度归一化
        reference_points_cam[..., 1] /= img_shape[0]  # 高度归一化
        
        # 15. 创建完整掩码（点在图像范围内）
        bev_mask = (
                    (reference_points_cam[..., 0] > 0.0) &
                    (reference_points_cam[..., 0] < 1.0) &
                    (reference_points_cam[..., 1] < 1.0) &
                    (reference_points_cam[..., 1] > 0.0))

        
        # 16. 处理NaN值（不同PyTorch版本兼容）
        # if digit_version(TORCH_VERSION) >= digit_version('1.8'):
        bev_mask = torch.nan_to_num(bev_mask)  # PyTorch 1.8+直接支持
        # else:
        #     bev_mask = bev_mask.new_tensor(
        #         np.nan_to_num(bev_mask.cpu().numpy()))  # 旧版本通过NumPy处理
        
        # 17. 调整维度顺序
        # 单摄像头优化：添加摄像头维度以保持接口兼容
        reference_points_cam = reference_points_cam.permute(0, 2, 1, 3).unsqueeze(0)
        # 新形状: [1, 1, 22500, 4, 2]
        
        # print('bev_mask',bev_mask.shape)
        bev_mask = bev_mask.permute(0,2,1).unsqueeze(0)
        # 新形状: [1, 1, 22500, 4]
        
        return reference_points_cam, bev_mask
    

    def forward(self,
                query,
                key,
                value,
                bev_h,
                bev_w,
                cam2img,
                spatial_shapes=None,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                reference_points_cam=None,
                bev_mask=None,
                flag='encoder',
                **kwargs):

        dtype = query.dtype
        bs, num_query, _ = query.size()

        # z_range=[1, 50],
        # x_range=[-25, 25],
        # y_range=[0, 3],
        
        self.pc_range = [-25 , 1, 0, 25, 50, 3.0]
        self.num_points_in_pillar = 4
        img_shape = self.img_size


        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, query.size(0), query.device, query.dtype)

        reference_points = ref_3d

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, cam2img, img_shape)


        mask_bev = torch.zeros((bs, bev_h, bev_w),
                               device=query.device).to(dtype)

        query_pos = self.position_embedding(mask_bev).to(dtype)
        query_pos_reshaped = query_pos.permute(0, 2, 3, 1)  # 变为 [16, 32, 32, 320]
        query_pos_reshaped = query_pos_reshaped.reshape(bs, self.feat_size_h*self.feat_size_w, 256)
        query_pos = query_pos_reshaped

        # print('query.shape',query.shape)
        # print('query_pos.shape',query_pos.shape)

        
        # 输入处理保持不变
        if key is None: key = query
        if value is None: value = key
        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        
        D = reference_points_cam.size(3)  # 保持不变
        
        # =====================================
        # 单摄像头优化点1: 简化索引处理
        # =====================================
        # 直接使用第一个摄像头的掩码（唯一摄像头）
        mask_per_img = bev_mask[0]
        # 获取有效查询索引
        index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
        max_len = len(index_query_per_img)  # 无需多摄像头取max
        
        # =====================================
        # 单摄像头优化点2: 避免冗余维度
        # =====================================
        # 创建查询缓存 (移除摄像头维度)
        queries_rebatch = query.new_zeros([bs, max_len, self.embed_dims])
        # 创建参考点缓存 (移除摄像头维度)
        reference_points_rebatch = reference_points_cam.new_zeros([bs, max_len, D, 2])
        
        # 仅处理单个摄像头的数据
        ref_points_per_img = reference_points_cam[0]  # 使用第一个摄像头的参考点
        
        # 填充查询和参考点
        for j in range(bs):
            # 只处理有效索引区域
            valid_len = min(max_len, len(index_query_per_img))
            queries_rebatch[j, :valid_len] = query[j, index_query_per_img[:valid_len]]
            reference_points_rebatch[j, :valid_len] = ref_points_per_img[j, index_query_per_img[:valid_len]]

        # =====================================
        # 单摄像头优化点3: 简化key/value处理
        # =====================================
        # 单摄像头时，key和value维度为 [1, seq_len, bs, dim]
        # 直接压缩掉摄像头维度
        # key = key[0]  # 现在形状: [seq_len, bs, dim]
        # value = value[0] # 现在形状: [seq_len, bs, dim]
        
        # 调整维度顺序以适应注意力模块: [bs, seq_len, dim]
        # key = key.permute(1, 0, 2)
        # value = value.permute(1, 0, 2)

        # =====================================
        # 单摄像头优化点4: 直接使用原始形状
        # =====================================
        # 执行可变形注意力计算
        attn_output = self.deformable_attention(
            query=queries_rebatch,  # [bs, max_len, dim]
            key=key,                # [bs, seq_len, dim]
            value=value,            # [bs, seq_len, dim]
            reference_points=reference_points_rebatch,  # [bs, max_len, D, 2]
            spatial_shapes=spatial_shapes
        )  # 输出形状: [bs, max_len, dim]

        # =====================================
        # 单摄像头优化点5: 简化结果聚合
        # =====================================
        for j in range(bs):
            # 直接将结果写回对应位置
            valid_len = min(max_len, len(index_query_per_img))
            slots[j, index_query_per_img[:valid_len]] = attn_output[j, :valid_len]

        # =====================================
        # 单摄像头优化点6: 简化覆盖计数
        # =====================================
        # 单摄像头覆盖情况直接使用bev_mask即可
        count = (bev_mask[0] > 0).sum(-1).squeeze(0)  # [bs, num_query]
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]

        # 输出投影和残差连接保持不变
        slots = self.output_proj(slots)



        # =====================================
        # 可视化
        # =====================================
        # features = self.dropout(slots)  # 替换为你的特征张量


        # # 创建保存目录
        # save_dir = '/opt/data/private/hwj_autodrive/UniCon/eval_metric_visual/featureviusal/Support_loss_datquery_fullnoise'
        # os.makedirs(save_dir, exist_ok=True)

        # batch_idx = 0    # 选择第一个批次
        # channel_idx = 0  # 选择第一个通道

        # # 转换为 numpy 数组并脱离梯度
        # features_np = features.detach().cpu().numpy()

        # # 提取指定批次和通道的特征序列 (N,)
        # sequence = features_np[batch_idx, :, channel_idx]

        # # 确定二维图像的尺寸 (H, W)
        # # 这里假设 N 可以被分解为 H * W，如果不能，可能需要调整
        # N = len(sequence)
        # H = int(np.sqrt(N))  # 高度
        # W = int(np.ceil(N / H))  # 宽度

        # # 创建二维图像
        # # 如果序列长度不是 H*W，用零填充
        # if H * W > N:
        #     padded_sequence = np.pad(sequence, (0, H*W - N), mode='constant')
        # else:
        #     padded_sequence = sequence[:H*W]

        # # 重塑为二维图像 (H, W)
        # image_2d = padded_sequence.reshape(H, W)

        # # 归一化到 0-255 范围
        # min_val = image_2d.min()
        # max_val = image_2d.max()
        # if max_val - min_val > 1e-6:
        #     image_2d = (image_2d - min_val) / (max_val - min_val)
        # image_2d = (image_2d * 255).astype(np.uint8)

        # # 应用颜色映射
        # colored = cv2.applyColorMap(image_2d, cv2.COLORMAP_JET)

        # # 创建保存目录

        # timestamp = int(time.time() * 1000)  # 毫秒级时间戳
        # save_path = os.path.join(save_dir, f"batch_{batch_idx}_channel_{channel_idx}_{timestamp}.png")
        # cv2.imwrite(save_path, colored)

        # print(f"图像已保存到: {save_path}")
        # print(f"原始序列长度: {N}, 图像尺寸: {H}x{W}")






        return self.dropout(slots) + inp_residual




class DeformableAttention3D(BaseModule):
    """单摄像头专用版本，优化计算效率和内存使用"""
    
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_points=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        
        # ================= 参数验证 =================
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims必须能被num_heads整除,'
                             f'但得到 {embed_dims} 和 {num_heads}')
        
        # ================= 核心参数 =================
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.batch_first = batch_first
        
        # ================= 单摄像头优化设置 =================
        self.num_levels = 1  # 固定为1，不支持多尺度
        
        # ================= 模型层定义 =================
        # 采样偏移预测 (输出num_points * 2个坐标偏移)
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_points * 2)  # 单尺度优化
        
        # 注意力权重预测
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_points)  # 单尺度优化
                                           
        # 值投影层
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        
        # ================= 初始化权重 =================
        self.init_weights()
    
    def init_weights(self):
        """针对单摄像头场景的特殊初始化"""
        # 偏移量初始化为零
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        nn.init.constant_(self.sampling_offsets.bias.data, 0.)
        
        # 注意力权重初始化为均匀分布
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        
        # 值投影层Xavier初始化
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        
        # 输出投影层Xavier初始化
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias.data, 0.)
    
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                **kwargs):
        """
        前向传播优化版，专为单摄像头设计
        
        参数简化：
        - spatial_shapes: 应为单元素列表 [[H, W]]
        - reference_points: 形状为 (bs, num_query, num_Z_anchors, 2)
        - level_start_index: 可忽略或设为[0]
        """
        
        # ================= 输入处理 =================
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos  # 添加位置编码
            
        # 维度转换 (适配批处理优先配置)
        if not self.batch_first:
            query = query.permute(1, 0, 2)  # (bs, num_query, dim)
            value = value.permute(1, 0, 2)
            
        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape
        
        # ================= 单尺度验证 =================
        # 确保单尺度特征图
        if spatial_shapes is None:
            # 如果没有提供，假设单尺度特征图
            spatial_shapes = torch.tensor([[value.size(1)]], 
                                         device=value.device, 
                                         dtype=torch.long)
        # elif spatial_shapes.nelement() != 2:
        #     # 确保只有一个层级
        #     assert len(spatial_shapes) == 1, "只支持单尺度特征图"
        #     spatial_shapes = spatial_shapes.view(1, 2)
        
        # ================= 值投影 =================
        value = self.value_proj(value)
        
        # 应用键值掩码 (如果提供)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        
        # 多头值分割 (避免视图操作)
        head_dim = self.embed_dims // self.num_heads
        value = value.view(bs, num_value, self.num_heads, head_dim)
        
        # ================= 偏移预测 =================
        # 预测采样偏移量 (每个点二维坐标)
        # 形状: (bs, num_query, num_heads * num_points * 2)
        sampling_offsets = self.sampling_offsets(query)
        
        # 重塑为: (bs, num_query, num_heads, num_points, 2)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        
        # ================= 注意力权重 =================
        # 预测注意力权重 (bs, num_query, num_heads*num_points)
        attention_weights = self.attention_weights(query)
        
        # 应用softmax归一化 (单尺度)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_points)
        attention_weights = attention_weights.softmax(dim=-1)
        
        # ================= 采样位置计算 =================
        # 简化参考点处理
        # reference_points形状: (bs, num_query, num_Z_anchors, 2)
        num_Z_anchors = reference_points.size(2)
        
        # 参考点标准化 (扩展到完整采样点集)
        reference_points = reference_points[:, :, None, None, None, :, :]  # (bs, nq, 1, Z, 2)
        # print('reference_points.shape',reference_points.shape)
        
        sampling_offsets = sampling_offsets.view(
                bs, num_query, self.num_heads, self.num_levels, self.num_points // num_Z_anchors, num_Z_anchors, 2)
        # print('sampling_offsets.shape',sampling_offsets.shape)
        sampling_locations = reference_points + sampling_offsets
        # print('sampling_locations.shape',sampling_locations.shape)
        
        # 重塑为期望格式: (bs, nq, heads, levels, points, Z, 2)
        sampling_locations = sampling_locations.view(
            bs, num_query, self.num_heads, 1, self.num_points, 2)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, 1, self.num_points)
        
        # ================= 可变形注意力计算 (纯PyTorch) =================
        # 调用PyTorch实现的可变形注意力
        output = multi_scale_deformable_attn_pytorch(
            value, 
            spatial_shapes,
            sampling_locations,
            attention_weights
        )
        
        # ================= 输出处理 =================
        # 合并多头输出
        output = output.view(bs, num_query, self.embed_dims)
        
        # 应用输出投影
        output = self.output_proj(output)
        
        # 恢复维度顺序
        if not self.batch_first:
            output = output.permute(1, 0, 2)
            
        return output 