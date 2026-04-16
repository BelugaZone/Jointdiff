import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_

class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class CrossDAttention(nn.Module):
    def __init__(
            self, channel, q_size, kv_size=None, n_heads=8, n_groups=4,
            attn_drop=0.0, proj_drop=0.0, stride=1,
            offset_range_factor=4, use_pe=True, dwc_pe=True,
            no_off=False, fixed_pe=False, ksize=3, log_cpb=False,
    ):
        """
        Cross Deformable Attention
        
        参数:
        - channel: 特征通道数
        - q_size: 查询特征图尺寸 (H_q, W_q)
        - kv_size: 键值特征图尺寸 (H_kv, W_kv) [可选]
        - n_heads: 注意力头数
        - n_groups: 分组数
        - stride: 键值下采样步长
        """
        super().__init__()
        
        # 查询特征图尺寸
        self.q_h, self.q_w = q_size
        
        # 键值特征图尺寸（动态计算或手动指定）
        if kv_size is None:
            self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        else:
            self.kv_h, self.kv_w = kv_size
        
        # 基本参数设置
        self.n_head_channels = channel // n_heads
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.nc = n_heads * self.n_head_channels
        self.n_groups = n_groups
        self.n_group_channels = self.nc // n_groups
        self.n_group_heads = n_heads // n_groups
        self.stride = stride
        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.fixed_pe = fixed_pe
        
        # 偏移量预测网络
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        # 查询投影层（从查询特征生成Q）
        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        
        # 键值投影层（从键值特征生成K和V）
        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        # 输出投影层
        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        # Dropout层
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        # 位置编码
        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                # 从Swin-V2借鉴的CPB编码
                self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_group_heads, bias=False)
                )
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        """生成参考点网格 (归一化到[-1, 1])"""
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)  # 归一化x坐标
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)  # 归一化y坐标
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B*g H W 2
        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):
        """生成查询网格 (归一化到[-1, 1])"""
        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)  # 归一化x坐标
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)  # 归一化y坐标
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B*g H W 2
        return ref

    def forward(self, q_feat, kv_feat):
        """
        前向传播
        参数:
        - q_feat: 查询特征 (B, C, H_q, W_q)
        - kv_feat: 键值特征 (B, C, H_kv, W_kv)
        """
        # 基础参数获取
        B, C, H_q, W_q = q_feat.size()
        _, _, H_kv, W_kv = kv_feat.size()
        dtype, device = q_feat.dtype, q_feat.device
        
        # 查询投影 (从查询特征图)
        q = self.proj_q(q_feat)
        
        # 键值特征图处理
        if self.kv_h != H_kv or self.kv_w != W_kv:
            # 调整键值特征图尺寸
            kv_feat = F.interpolate(
                kv_feat, 
                size=(self.kv_h, self.kv_w), 
                mode='bilinear', 
                align_corners=True
            )
            H_kv, W_kv = self.kv_h, self.kv_w
        
        # 预测偏移量 (从查询特征)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', 
                                g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B*g 2 H_q' W_q'
        n_sample = offset.size(2) * offset.size(3)
        
        # 限制偏移范围
        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor(
                [1.0 / (offset.size(2) - 1.0), 1.0 / (offset.size(3) - 1.0)], 
                device=device
            ).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
        
        # 准备采样位置
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(
            offset.size(1), offset.size(2), B, dtype, device
        )  # B*g H_q' W_q' 2
        
        if self.no_off:
            offset = offset.fill_(0.0)
            
        if self.offset_range_factor >= 0:
            pos = offset + reference  # 偏移位置
        else:
            pos = (offset + reference).clamp(-1., +1.)
        
        # 变形采样键值特征
        if self.no_off:
            # 直接使用键值特征 (不下采样)
            x_sampled = kv_feat
        else:
            # 可变形采样
            x_sampled = F.grid_sample(
                input=kv_feat.reshape(B * self.n_groups, self.n_group_channels, H_kv, W_kv),
                grid=pos[..., (1, 0)],  # y,x -> x,y
                mode='bilinear', 
                align_corners=True
            )  # B*g C_g H_q' W_q'
        
        # 重组采样特征
        x_sampled = x_sampled.reshape(B, C, offset.size(1), offset.size(2))
        x_sampled = einops.rearrange(x_sampled, 'b c h w -> b c (h w)')
        
        # 准备Q, K, V
        q = einops.rearrange(q, 'b c h w -> b c (h w)')
        k = self.proj_k(x_sampled)
        v = self.proj_v(x_sampled)
        
        # 多头注意力拆分
        q = einops.rearrange(q, 'b (h c) n -> b h c n', h=self.n_heads)
        k = einops.rearrange(k, 'b (h c) n -> b h c n', h=self.n_heads)
        v = einops.rearrange(v, 'b (h c) n -> b h c n', h=self.n_heads)
        
        # 注意力计算: Q^T K / sqrt(d)
        attn = torch.einsum('b h c m, b h c n -> b h m n', q, k)
        attn = attn.mul(self.scale)
        
        # 位置编码添加
        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                # 深度卷积位置编码
                residual_lepe = self.rpe_table(q_feat).reshape(
                    B * self.n_heads, self.n_head_channels, H_q * W_q
                )
            elif self.fixed_pe:
                # 固定位置编码
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H_q * W_q, n_sample)
            elif self.log_cpb:
                # 相对位置对数偏置
                q_grid = self._get_q_grid(H_q, W_q, B, dtype, device)
                displacement = (
                    q_grid.reshape(B * self.n_groups, H_q * W_q, 2).unsqueeze(2) - 
                    pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)
                ).mul(4.0)  # d_y, d_x [-8, +8]
                displacement = torch.sign(displacement) * torch.log2(
                    torch.abs(displacement) + 1.0
                ) / np.log2(8.0)
                attn_bias = self.rpe_table(displacement)
                attn = attn + einops.rearrange(
                    attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads
                )
            else:
                # 相对位置编码
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                q_grid = self._get_q_grid(H_q, W_q, B, dtype, device)
                displacement = (
                    q_grid.reshape(B * self.n_groups, H_q * W_q, 2).unsqueeze(2) - 
                    pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)
                ).mul(0.5)
                attn_bias = F.grid_sample(
                    input=einops.rearrange(
                        rpe_bias, 'b (g c) h w -> (b g) c h w', 
                        c=self.n_group_heads, g=self.n_groups
                    ),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', 
                    align_corners=True
                )
                attn_bias = attn_bias.reshape(B * self.n_heads, H_q * W_q, n_sample)
                attn = attn + attn_bias
        
        # 注意力权重和输出计算
        attn = F.softmax(attn, dim=3)
        attn = self.attn_drop(attn)
        out = torch.einsum('b h m n, b h c n -> b h c m', attn, v)
        
        # 添加位置编码残差
        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe.unsqueeze(0)
        
        # 多头特征合并
        out = einops.rearrange(out, 'b h c m -> b (h c) m')
        out = einops.rearrange(out, 'b c (h w) -> b c h w', h=H_q, w=W_q)
        
        # 输出投影
        out = self.proj_out(out)
        out = self.proj_drop(out)
        
        return out


if __name__ == '__main__':
    # 测试交叉注意力
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 查询特征 (B, C, H_q, W_q)
    q_feat = torch.randn(4, 512, 14, 14).to(device)
    
    # 键值特征 (B, C, H_kv, W_kv) - 尺寸不同
    kv_feat = torch.randn(4, 512, 28, 28).to(device)
    
    # 初始化交叉注意力
    model = CrossDAttention(
        channel=512,
        q_size=(14, 14),
        kv_size=(7, 7),  # 设置键值采样尺寸
        n_heads=8,
        n_groups=4,
        stride=2,
        offset_range_factor=4
    ).to(device)
    
    # 前向传播
    out = model(q_feat, kv_feat)
    
    print("输入查询尺寸:", q_feat.shape)
    print("输入键值尺寸:", kv_feat.shape)
    print("输出尺寸:", out.shape)  # 应保持与查询特征相同尺寸