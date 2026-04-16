import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List


ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class BEVEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        bev_height: int,
        bev_width: int,
        h_meters: int,
        w_meters: int,
        offset: int,
        h: int,
        w: int,
        
    ):
        """
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()

        # each decoder block upsamples the bev embedding by a factor of 2
        # h = 32
        # w = 32

        # bev coordinates
        grid = generate_grid(h, w).squeeze(0)
        grid[0] = bev_width * grid[0]
        grid[1] = bev_height * grid[1]

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()                                  # 3 3
        grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')                      # 3 (h w)
        grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)                    # 3 h w

        # egocentric frame
        self.register_buffer('grid', grid, persistent=False)                    # 3 h w
    #     self.learned_features = nn.Parameter(sigma * torch.randn(dim, h, w))    # d h w

    # def get_prior(self):
    #     return self.learned_features


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias=True, norm=nn.LayerNorm):
        """
        完整功能单视角交叉注意力
        dim: 特征维度
        heads: 多头数量
        dim_head: 每个头的维度
        qkv_bias: 是否使用偏置
        norm: 归一化层类型
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        # 完整的QKV变换（带归一化）
        self.to_q = nn.Sequential(
            norm(dim),
            nn.Linear(dim, heads * dim_head, bias=qkv_bias)
        )
        self.to_k = nn.Sequential(
            norm(dim),
            nn.Linear(dim, heads * dim_head, bias=qkv_bias)
        )
        self.to_v = nn.Sequential(
            norm(dim),
            nn.Linear(dim, heads * dim_head, bias=qkv_bias)
        )

        # 特征增强模块
        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim)
        )
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None):
        """
        完整单视角交叉注意力前向传播
        q: (b, d, H, W)  [BEV查询特征]
        k: (b, d, h, w)  [图像键特征]
        v: (b, d, h, w)  [图像值特征]
        skip: (b, d, H, W) 可选残差连接

        返回: (b, d, H, W)  [更新后的BEV特征]
        """
        b, d, H, W = q.shape
        _, _, h, w = k.shape

        # 展平空间维度 (保持BEV和图像空间分离)
        q_flat = rearrange(q, 'b d H W -> b (H W) d')  # (b, H*W, d)
        k_flat = rearrange(k, 'b d h w -> b (h w) d')  # (b, h*w, d)
        v_flat = rearrange(v, 'b d h w -> b (h w) d')  # (b, h*w, d)

        # 投影到查询、键、值
        q_proj = self.to_q(q_flat)  # (b, H*W, heads*dim_head)
        k_proj = self.to_k(k_flat)  # (b, h*w, heads*dim_head)
        v_proj = self.to_v(v_flat)  # (b, h*w, heads*dim_head)

        # 分离多头
        q = rearrange(q_proj, 'b n (h d) -> b h n d', h=self.heads, d=self.dim_head)  # (b, heads, H*W, dim_head)
        k = rearrange(k_proj, 'b m (h d) -> b h m d', h=self.heads, d=self.dim_head)  # (b, heads, h*w, dim_head)
        v = rearrange(v_proj, 'b m (h d) -> b h m d', h=self.heads, d=self.dim_head)  # (b, heads, h*w, dim_head)

        # 注意力计算
        attn = torch.einsum('b h n d, b h m d -> b h n m', q, k) * self.scale  # (b, heads, H*W, h*w)
        attn = attn.softmax(dim=-1)

        # 聚合值
        out = torch.einsum('b h n m, b h m d -> b h n d', attn, v)  # (b, heads, H*W, dim_head)

        # 合并多头
        out = rearrange(out, 'b h n d -> b n (h d)')  # (b, H*W, heads*dim_head)
        z = self.proj(out)  # (b, H*W, d)

        # 残差连接
        if skip is not None:
            skip_flat = rearrange(skip, 'b d H W -> b (H W) d')
            z = z + skip_flat

        # 特征增强
        z = self.prenorm(z)
        z = self.mlp(z) + z  # 残差连接
        z = self.postnorm(z)

        # 恢复空间结构
        # z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z


class CrossViewAttention(nn.Module):
    def __init__(
            self,
            feat_height: int,
            feat_width: int,
            feat_dim: int,
            dim: int,
            image_height: int,
            image_width: int,
            qkv_bias: bool,
            heads: int = 4,
            dim_head: int = 32,
            no_image_features: bool = False,
            skip: bool = True,
    ):
        super().__init__()

        # 生成单摄像头图像平面网格 (1x1x3xhxw)
        image_plane = generate_grid(feat_height, feat_width)
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height
        self.register_buffer('image_plane', image_plane, persistent=False)

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False))

        # BEV和图像编码器
        self.bev_embed = nn.Conv2d(2, dim, 1)  # 处理2D BEV坐标
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)  # 处理3D空间位置
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)  # 相机位置编码

        # 单摄像头交叉注意力机制
        self.cross_attend_bevQ = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.cross_attend_imgQ = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

    def forward(
            self,
            x: torch.FloatTensor,
            bev: BEVEmbedding,
            feature: torch.FloatTensor,
            I: torch.FloatTensor,
            E: torch.FloatTensor,
    ):
        """
        修改为单摄像头输入:
        x: (b, c, H, W)      - BEV特征
        feature: (b, dim_in, h, w) - 单摄像头图像特征
        I_inv: (b, 3, 3)      - 单摄像头内参逆矩阵
        E_inv: (b, 4, 4)      - 单摄像头外参逆矩阵

        Returns: (b, d, H, W) - 更新后的BEV特征
        """
        b = feature.shape[0]
        _, _, h, w = self.image_plane.shape
        E_inv = E.inverse()
        I_inv = I.inverse()
        # 相机位置提取 (从外参矩阵获取)
        c = E_inv[..., -1]  # 提取平移分量 (b, 4) 而不是 (b, 4, 1)
        # 添加缺失的维度
        c = c.unsqueeze(-1).unsqueeze(-1) 
        c_embed = self.cam_embed(c)  # (b, d, 1, 1)

        # 生成图像平面坐标并投影到3D空间
        pixel_flat = rearrange(self.image_plane, '... h w -> ... (h w)')  # (1, 3, h*w)
        cam = I_inv @ pixel_flat  # (b, 3, h*w)
        cam = F.pad(cam, (0, 0, 0, 1), value=1)  # (b, 4, h*w) 添加齐次坐标
        d = E_inv @ cam  # (b, 4, h*w) 世界坐标系中的3D点

        # 空间位置编码
        d_flat = rearrange(d, 'b d (h w) -> b d h w', h=h, w=w)  # (b, 4, h, w)
        d_embed = self.img_embed(d_flat)  # (b, d, h, w)

        # 相对位置编码 (图像位置 - 相机位置)
        img_embed = d_embed - c_embed  # (b, d, h, w)
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)

        # BEV位置编码
        world = bev.grid[:2]  # (2, H, W) BEV网格的XY坐标
        w_embed = self.bev_embed(world[None])  # (1, d, H, W)

        # 相对位置编码 (BEV位置 - 相机位置)
        bev_embed = w_embed - c_embed  # (b, d, H, W)
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
        query_pos = bev_embed  # (b, d, H, W) 直接作为查询位置编码

        # # 图像特征处理
        # if self.feature_proj is not None:
        #     key = img_embed + self.feature_proj(feature)  # (b, d, h, w)
        # else:
        #     key = img_embed  # (b, d, h, w)

        # val = self.feature_linear(feature)  # (b, d, h, w)

        # # 准备查询向量 (BEV位置编码 + 当前BEV特征)
        # query = query_pos + x  # (b, d, H, W)
         



        if self.feature_proj is not None:
            img_embed_proj = img_embed + self.feature_proj(feature)  # (b, d, h, w)
        else:
            key = img_embed  # (b, d, h, w)

        img_feature = self.feature_linear(feature)  # (b, d, h, w)

        # 准备查询向量 (BEV位置编码 + 当前BEV特征)
        bev_feature = x
        

        # return img_feature , bev_feature , img_embed , bev_embed
        bevQ_query = bev_embed + bev_feature
        bevQ_key = img_embed + img_feature
        bevQ_v = img_feature

        imgQ_query = img_embed + img_feature
        imgQ_key = bev_embed + bev_feature
        imgQ_v = bev_feature

 



        # 执行单摄像头交叉注意力
        feature_bev =  self.cross_attend_bevQ(
            bevQ_query,
            bevQ_key,
            bevQ_v,
            skip=x if self.skip else None
        )

        feature_img =  self.cross_attend_imgQ(
            imgQ_query,
            imgQ_key,
            imgQ_v,
            skip=x if self.skip else None
        )

        return feature_img, feature_bev
# class CrossViewAttention_reverse(nn.Module):
#     def __init__(
#             self,
#             feat_height: int,
#             feat_width: int,
#             feat_dim: int,
#             dim: int,
#             image_height: int,
#             image_width: int,
#             qkv_bias: bool,
#             heads: int = 4,
#             dim_head: int = 32,
#             no_image_features: bool = False,
#             skip: bool = True,
#     ):
#         super().__init__()

#         # 生成单摄像头图像平面网格 (1x1x3xhxw)
#         image_plane = generate_grid(feat_height, feat_width)
#         image_plane[:, :, 0] *= image_width
#         image_plane[:, :, 1] *= image_height
#         self.register_buffer('image_plane', image_plane, persistent=False)

#         self.feature_linear = nn.Sequential(
#             nn.BatchNorm2d(feat_dim),
#             nn.ReLU(),
#             nn.Conv2d(feat_dim, dim, 1, bias=False))

#         if no_image_features:
#             self.feature_proj = None
#         else:
#             self.feature_proj = nn.Sequential(
#                 nn.BatchNorm2d(feat_dim),
#                 nn.ReLU(),
#                 nn.Conv2d(feat_dim, dim, 1, bias=False))

#         # BEV和图像编码器
#         self.bev_embed = nn.Conv2d(2, dim, 1)  # 处理2D BEV坐标
#         self.img_embed = nn.Conv2d(4, dim, 1, bias=False)  # 处理3D空间位置
#         self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)  # 相机位置编码

#         # 单摄像头交叉注意力机制
#         self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)
#         self.skip = skip

#     def forward(
#             self,
#             x: torch.FloatTensor,
#             bev: BEVEmbedding,
#             feature: torch.FloatTensor,
#             I: torch.FloatTensor,
#             E: torch.FloatTensor,
#     ):
#         """
#         修改为单摄像头输入:
#         x: (b, c, H, W)      - BEV特征
#         feature: (b, dim_in, h, w) - 单摄像头图像特征
#         I_inv: (b, 3, 3)      - 单摄像头内参逆矩阵
#         E_inv: (b, 4, 4)      - 单摄像头外参逆矩阵

#         Returns: (b, d, H, W) - 更新后的BEV特征
#         """
#         b = feature.shape[0]
#         _, _, h, w = self.image_plane.shape
#         E_inv = E.inverse()
#         I_inv = I.inverse()
#         # 相机位置提取 (从外参矩阵获取)
#         c = E_inv[..., -1]  # 提取平移分量 (b, 4) 而不是 (b, 4, 1)
#         # 添加缺失的维度
#         c = c.unsqueeze(-1).unsqueeze(-1) 
#         c_embed = self.cam_embed(c)  # (b, d, 1, 1)

#         # 生成图像平面坐标并投影到3D空间
#         pixel_flat = rearrange(self.image_plane, '... h w -> ... (h w)')  # (1, 3, h*w)
#         cam = I_inv @ pixel_flat  # (b, 3, h*w)
#         cam = F.pad(cam, (0, 0, 0, 1), value=1)  # (b, 4, h*w) 添加齐次坐标
#         d = E_inv @ cam  # (b, 4, h*w) 世界坐标系中的3D点

#         # 空间位置编码
#         d_flat = rearrange(d, 'b d (h w) -> b d h w', h=h, w=w)  # (b, 4, h, w)
#         d_embed = self.img_embed(d_flat)  # (b, d, h, w)

#         # 相对位置编码 (图像位置 - 相机位置)
#         img_embed = d_embed - c_embed  # (b, d, h, w)
#         img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)

#         # BEV位置编码
#         world = bev.grid[:2]  # (2, H, W) BEV网格的XY坐标
#         w_embed = self.bev_embed(world[None])  # (1, d, H, W)

#         # 相对位置编码 (BEV位置 - 相机位置)
#         bev_embed = w_embed - c_embed  # (b, d, H, W)
#         bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
#         query_pos = bev_embed  # (b, d, H, W) 直接作为查询位置编码

#         # 图像特征处理
#         if self.feature_proj is not None:
#             key = query_pos + x  # (b, d, h, w)
#         else:
#             key = query_pos + x  # (b, d, h, w)

#         val = x  # (b, d, h, w)

#         # 准备查询向量 (BEV位置编码 + 当前BEV特征)
#         query = img_embed + self.feature_proj(feature)  # (b, d, H, W)
         

        
#         # 执行单摄像头交叉注意力
#         return self.cross_attend(
#             query,
#             key,
#             val,
#             skip=x if self.skip else None
#         )


class Encoder(nn.Module):
    def __init__(
            self,
            backbone,
            cross_view: dict,
            bev_embedding: dict,
            dim: int = 128,
            middle: List[int] = [2, 2],
            scale: float = 1.0,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone

        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)

        cross_views = list()
        layers = list()

        for feat_shape, num_layers in zip(self.backbone.output_shapes, middle):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            cva = CrossViewAttention(feat_height, feat_width, feat_dim, dim, **cross_view)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)])
            layers.append(layer)

        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)            # b n c h w
        I_inv = batch['intrinsics'].inverse()           # b n 3 3
        E_inv = batch['extrinsics'].inverse()           # b n 4 4

        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()              # d H W
        x = repeat(x, '... -> b ...', b=b)              # b d H W

        for cross_view, feature, layer in zip(self.cross_views, features, self.layers):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = cross_view(x, self.bev_embedding, feature, I_inv, E_inv)
            x = layer(x)

        return x
