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
        # self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)
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

        # 图像特征处理
        if self.feature_proj is not None:
            img_embed_proj = img_embed + self.feature_proj(feature)  # (b, d, h, w)
        else:
            key = img_embed  # (b, d, h, w)

        img_feature = self.feature_linear(feature)  # (b, d, h, w)

        # 准备查询向量 (BEV位置编码 + 当前BEV特征)
        bev_feature = x
        

        return img_feature , bev_feature , img_embed , bev_embed

        # # 执行单摄像头交叉注意力
        # return self.cross_attend(
        #     query,
        #     key,
        #     val,
        #     skip=x if self.skip else None
        # )




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
