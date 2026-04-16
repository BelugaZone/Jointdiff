from functools import partial
import math
from typing import Iterable
from torch import nn, einsum
import numpy as np
import torch as th
import torch.nn as nn
import functools
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List, Callable, Dict
import math
from einops import rearrange, repeat
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange
import copy
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models.resnet import resnet18
from .deformable_atten import SpatialCrossAttention
import cv2
import os
import time

coco_category_list_check_person = [
    "arm",
    'person',
    "man",
    "woman",
    "child",
    "boy",
    "girl",
    "teenager"
]

VOC_category_list_check = {
    'aeroplane': ['aerop', 'lane'],
    'bicycle': ['bicycle'],
    'bird': ['bird'],
    'boat': ['boat'],
    'bottle': ['bottle'],
    'bus': ['bus'],
    'car': ['car'],
    'cat': ['cat'],
    'chair': ['chair'],
    'cow': ['cow'],
    'diningtable': ['table'],
    'dog': ['dog'],
    'horse': ['horse'],
    'motorbike': ['motorbike'],
    'person': coco_category_list_check_person,
    'pottedplant': ['pot', 'plant', 'ted'],
    'sheep': ['sheep'],
    'sofa': ['sofa'],
    'train': ['train'],
    'tvmonitor': ['monitor', 'tv', 'monitor']
}


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def resize_fn(img, size):
    return transforms.Resize(size, InterpolationMode.BICUBIC)(
        transforms.ToPILImage()(img))


import math


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# Projection of x onto y
def proj(x, y):
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x


def power_iteration(W, u_, update=True, eps=1e-12):
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        # Run one step of the power iteration
        with torch.no_grad():
            v = torch.matmul(u, W)
            # Run Gram-Schmidt to subtract components of all other singular vectors
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            # Add to the list
            vs += [v]
            # Update the other singular vector
            u = torch.matmul(v, W.t())
            # Run Gram-Schmidt to subtract components of all other singular vectors
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            # Add to the list
            us += [u]
            if update:
                u_[i][:] = u
        # Compute this singular value and add it to the list
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
        # svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
    return svs, us, vs


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)  # 相当与通道维度上连接，以弥补因为使用mb导致的卷积信息丢失。
        return self.conv(x1)


class SegEncode(nn.Module):
    def __init__(self, inC, outC, size):
        super().__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu
        self.up_sampler = nn.Upsample(size=size, mode='bilinear', align_corners=True)
        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
            # nn.Sigmoid(),
        )

    def forward(self, x):  # torch.Size([2, 256, 200, 400])
        x = self.up_sampler(x)
        x = self.conv1(x)  # torch.Size([2, 64, 200, 400])
        x = self.bn1(x)
        x = self.relu(x)
        # import pdb; pdb.set_trace()
        x1 = self.layer1(x)  # torch.Size([2, 64, 100, 200])
        x = self.layer2(x1)  # torch.Size([2, 128, 50, 100])
        x2 = self.layer3(x)  # torch.Size([2, 256, 25, 50])

        x = self.up1(x2, x1)  # torch.Size([2, 256, 100, 200])
        x = self.up2(x)  # torch.Size([2, 4, 200, 400]) 语义分割预测特征图

        return x


# Spectral normalization base class
class SN(object):
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        # Number of power iterations per step
        self.num_itrs = num_itrs
        # Number of singular values
        self.num_svs = num_svs
        # Transposed?
        self.transpose = transpose
        # Epsilon value for avoiding divide-by-0
        self.eps = eps
        # Register a singular vector for each sv
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    # Singular vectors (u side)
    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    # Singular values;
    # note that these buffers are just for logging and are not used in training.
    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    # Compute the spectrally-normalized weight
    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        # Apply num_itrs power iterations
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
            # Update the svs
        if self.training:
            with torch.no_grad():  # Make sure to do this in a no_grad() context or you'll get memory leaks!
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
    def __init__(self, in_features, out_features, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)

    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)


# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class SegBlock(nn.Module):
    def __init__(self, in_channels, out_channels, con_channels,
                 which_conv=nn.Conv2d, which_linear=None, activation=None,
                 upsample=None):
        super(SegBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_linear = which_conv, which_linear
        self.activation = activation
        self.upsample = upsample

        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)

        self.register_buffer('stored_mean1', torch.zeros(in_channels))
        self.register_buffer('stored_var1', torch.ones(in_channels))
        self.register_buffer('stored_mean2', torch.zeros(out_channels))
        self.register_buffer('stored_var2', torch.ones(out_channels))

        self.upsample = upsample

    def forward(self, x, y=None):
        x = F.batch_norm(x, self.stored_mean1, self.stored_var1, None, None,
                         self.training, 0.1, 1e-4)
        h = self.activation(x)
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = F.batch_norm(h, self.stored_mean2, self.stored_var2, None, None,
                         self.training, 0.1, 1e-4)

        h = self.activation(h)
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


def aggregate_attention(attention_store, res: int, from_where: List[str], is_cross: bool, select: int, prompts=None):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])
                #                 print(cross_maps.shape)
                out.append(cross_maps)

    out = torch.cat(out, dim=1)
    #     print(out.shape)
    return out


class seg_decorder(nn.Module):

    def __init__(self,
                 out_size=(256, 256),
                 hidden_dim=256,
                 num_classes=19,
                 mask_dim=256,
                 dim_feedforward=2048):
        super().__init__()

        # self.num_queries = num_queries
        # # learnable query features
        # self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # # learnable query p.e.
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.bev_embedding = nn.Embedding(200 * 200,256)
        self.out_size = out_size
        self.seg_head = SegEncode(256, 6, self.out_size)
        self.seg_head_img = SegEncode(256, 6, self.out_size)
        self.crossview_bevformer = SpatialCrossAttention(
                feat_size_h=200,
                feat_size_w=200,
                img_size=(256,256)

            )

        # self.query_feat_mlp = nn.Linear(hidden_dim + 768, hidden_dim, bias=False)
        # self.query_embed_mlp = nn.Linear(hidden_dim + 768, hidden_dim, bias=False)

        # self.decoder_norm = nn.LayerNorm(hidden_dim)

        # # positional encoding
        # N_steps = hidden_dim // 2

        # # level embedding (we always use 3 scales)
        # self.num_feature_levels = 3
        # self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        # self.input_proj = nn.ModuleList()

        # for _ in range(self.num_feature_levels):
        #     self.input_proj.append(nn.Sequential())

        # output FFNs
        # self.mask_classification = True
        # if self.mask_classification:
        #     self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # define Transformer decoder here
        # self.num_heads = 8
        # self.num_layers = 10
        # self.transformer_self_attention_layers = nn.ModuleList()
        # self.transformer_cross_attention_layers = nn.ModuleList()
        # self.transformer_ffn_layers = nn.ModuleList()

        # pre_norm = False
        # for _ in range(self.num_layers):
        #     self.transformer_self_attention_layers.append(
        #         SelfAttentionLayer(
        #             d_model=hidden_dim,
        #             nhead=self.num_heads,
        #             dropout=0.0,
        #             normalize_before=pre_norm,
        #         )
        #     )

        #     self.transformer_cross_attention_layers.append(
        #         CrossAttentionLayer(
        #             d_model=hidden_dim,
        #             nhead=self.num_heads,
        #             dropout=0.0,
        #             normalize_before=pre_norm,
        #         )
        #     )

        #     self.transformer_ffn_layers.append(
        #         FFNLayer(
        #             d_model=hidden_dim,
        #             dim_feedforward=dim_feedforward,
        #             dropout=0.0,
        #             normalize_before=False,
        #         )
        #     )

        low_feature_channel = 256
        mid_feature_channel = 256
        high_feature_channel = 256
        highest_feature_channel = 256

        self.low_feature_conv = nn.Sequential(
            nn.Conv2d(5120, low_feature_channel, kernel_size=1, bias=False),

        )
        self.mid_feature_conv = nn.Sequential(
            nn.Conv2d(4480, mid_feature_channel, kernel_size=1, bias=False),

        )
        self.mid_feature_mix_conv = SegBlock(
            in_channels=low_feature_channel + mid_feature_channel,
            out_channels=mask_dim,
            con_channels=128,
            which_conv=functools.partial(SNConv2d,
                                         kernel_size=3, padding=1,
                                         num_svs=1, num_itrs=1,
                                         eps=1e-04),
            which_linear=functools.partial(SNLinear,
                                           num_svs=1, num_itrs=1,
                                           eps=1e-04),
            activation=nn.ReLU(inplace=True),
            upsample=False,
        )

        self.high_feature_conv = nn.Sequential(
            nn.Conv2d(2880, high_feature_channel, kernel_size=1, bias=False),
        )
        self.high_feature_mix_conv = SegBlock(
            in_channels=mask_dim + high_feature_channel,
            out_channels=mask_dim,
            con_channels=128,
            which_conv=functools.partial(SNConv2d,
                                         kernel_size=3, padding=1,
                                         num_svs=1, num_itrs=1,
                                         eps=1e-04),
            which_linear=functools.partial(SNLinear,
                                           num_svs=1, num_itrs=1,
                                           eps=1e-04),
            activation=nn.ReLU(inplace=True),
            upsample=False,
        )
        self.highest_feature_conv = nn.Sequential(
            nn.Conv2d(1920, highest_feature_channel, kernel_size=1, bias=False),
        )
        self.highest_feature_mix_conv = SegBlock(
            in_channels=mask_dim + highest_feature_channel,
            out_channels=mask_dim,
            con_channels=128,
            which_conv=functools.partial(SNConv2d,
                                         kernel_size=3, padding=1,
                                         num_svs=1, num_itrs=1,
                                         eps=1e-04),
            which_linear=functools.partial(SNLinear,
                                           num_svs=1, num_itrs=1,
                                           eps=1e-04),
            activation=nn.ReLU(inplace=True),
            upsample=False,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

    # def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
    #     decoder_output = self.decoder_norm(output)
    #     decoder_output = decoder_output.transpose(0, 1)
    #     outputs_class = self.class_embed(decoder_output)
    #     mask_embed = self.mask_embed(decoder_output)
    #     outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
    #
    #     # NOTE: prediction is of higher-resolution
    #     # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
    #     attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
    #     # must use bool type
    #     # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
    #     attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
    #                                                                                                      1) < 0.5).bool()
    #     attn_mask = attn_mask.detach()
    #
    #     return outputs_class, outputs_mask, attn_mask

    def forward(self, diffusion_features,camera_intrinsic):

        x, all_features = self._prepare_features(diffusion_features)
        # print('all_features',all_features.shape)
        batch_size = all_features.size(0)
        start = batch_size // 2
        end = batch_size
        mask_features = all_features[start:end,:,:,:]  
        img_features = all_features[:start,:,:,:]  # [8, 256, 64, 64])
        # print('img_features',img_features.shape)
        dtype = all_features.dtype
        B, C, H, W = img_features.shape

        # 方法1: 使用 permute 和 reshape
        img_features_reshaped = img_features.permute(0, 2, 3, 1).reshape(B, H*W, C)

        # print('mask_features',mask_features.shape)

        out = self.seg_head(mask_features)

        bev_query = self.bev_embedding.weight.to(dtype)
        bev_query = bev_query.unsqueeze(0)  # 添加批次维度 [1, H*W, C]
        bev_query = bev_query.expand(start, -1, -1) 

        bev_feature_pred = self.crossview_bevformer(
                         bev_query,
                         img_features_reshaped,
                         img_features_reshaped,
                         200,
                         200,
                         camera_intrinsic,
                         [[64, 64]] ,

                    )      # [8, 40000, 256]

       



        # print('bev_feature_pred',bev_feature_pred.shape)
        bev_feature_2d = bev_feature_pred.view(B, 200, 200, C).permute(0, 3, 1, 2)
        img2bev_out = self.seg_head_img(bev_feature_2d)


         # # =====================================
        # # 可视化
        # # =====================================
        # feature_map = bev_feature_2d  # 替换为你的特征张量


        # # 创建保存目录
        # print('feature_map.shape',feature_map.shape)

        # feature_map = feature_map.detach().cpu().numpy()
        # save_dir = '/opt/data/private/hwj_autodrive/UniCon/eval_metric_visual/featureviusal/nomseloss_seghead/img_seghead'

        # # 处理批次维度
        # if len(feature_map.shape) == 4:  # [B, C, H, W]
        #     feature_map = feature_map[0]  # 取第一个批次

        # # 提取指定通道
        # channel_data = feature_map[0]

        # # 确保数据在0-255范围内
        # min_val = channel_data.min()
        # max_val = channel_data.max()
        # if max_val - min_val > 1e-6:  # 避免除以零
        #     channel_data = (channel_data - min_val) / (max_val - min_val)
        # channel_data = (channel_data * 255).astype(np.uint8)
        # channel_data = np.clip(channel_data, 0, 255).astype(np.uint8)

        # # 应用颜色映射（将灰度转换为彩色）
        # colored = cv2.applyColorMap(channel_data, cv2.COLORMAP_JET)

        # # 确保目录存在
        # os.makedirs(save_dir, exist_ok=True)

        # # 生成唯一文件名（使用时间戳）
        # timestamp = int(time.time() * 1000)  # 毫秒级时间戳
        # filename = f"feature_{timestamp}.png"  # 使用PNG格式
        # save_path = os.path.join(save_dir, filename)

        # # 保存图像
        # cv2.imwrite(save_path, colored)
        # print(f"图像已保存到: {save_path}")



        # print('out',out.shape)

        # out = {
        #     'pred_logits': predictions_class[-1],
        #     'pred_masks': predictions_mask[-1]
        # }
        return out , img2bev_out

    def _prepare_features(self, features, upsample='bilinear'):

        
        # self.low_feature_size = (8,14)
        # self.mid_feature_size = (16,28)
        # self.high_feature_size = (32,56)
        self.low_feature_size = (8,8)
        self.mid_feature_size = (16,16)
        self.high_feature_size = (32,32)

        self.final_high_feature_size = 64   #160origin

        low_features = [
            F.interpolate(i, size=self.low_feature_size, mode=upsample, align_corners=False) for i in features["low"]
        ]
        low_features = torch.cat(low_features, dim=1)
        # print('low_features.shape',low_features.shape)

        mid_features = [
            F.interpolate(i, size=self.mid_feature_size, mode=upsample, align_corners=False) for i in features["mid"]
        ]
        mid_features = torch.cat(mid_features, dim=1)
        # print('mid_features.shape',mid_features.shape)

        high_features = [
            F.interpolate(i, size=self.high_feature_size, mode=upsample, align_corners=False) for i in features["high"]
        ]
        high_features = torch.cat(high_features, dim=1)
        # print('high_features.shape',high_features.shape)

        highest_features = torch.cat(features["highest"], dim=1)
        # print('highest_features.shape',highest_features.shape)

        ## Attention map
        from_where = ("up", "down")
        select = 0

        # "up", "down"
        # attention_maps_8s = aggregate_attention(attention_store, 8, ("up", "mid", "down"), True, select,
        #                                         prompts=prompts)
        # attention_maps_16s = aggregate_attention(attention_store, 16, from_where, True, select, prompts=prompts)
        # attention_maps_32 = aggregate_attention(attention_store, 32, from_where, True, select, prompts=prompts)
        # attention_maps_64 = aggregate_attention(attention_store, 64, from_where, True, select, prompts=prompts)
        #
        # attention_maps_8s = rearrange(attention_maps_8s, 'b c h w d-> b (c d) h w')
        # attention_maps_16s = rearrange(attention_maps_16s, 'b c h w d-> b (c d) h w')
        # attention_maps_32 = rearrange(attention_maps_32, 'b c h w d-> b (c d) h w')
        # attention_maps_64 = rearrange(attention_maps_64, 'b c h w d-> b (c d) h w')
        #
        # attention_maps_8s = F.interpolate(attention_maps_8s, size=self.low_feature_size, mode=upsample,
        #                                   align_corners=False)
        #
        # attention_maps_16s = F.interpolate(attention_maps_16s, size=self.mid_feature_size, mode=upsample,
        #                                    align_corners=False)
        # attention_maps_32 = F.interpolate(attention_maps_32, size=self.high_feature_size, mode=upsample,
        #                                   align_corners=False)

        # features_dict = {
        #     'low': torch.cat([low_features, attention_maps_8s], dim=1),
        #     'mid': torch.cat([mid_features, attention_maps_16s], dim=1),
        #     'high': torch.cat([high_features, attention_maps_32], dim=1),
        #     'highest': torch.cat([highest_features, attention_maps_64], dim=1),
        # }

        features_dict = {
            'low': low_features.float().to(self.device),
            'mid': mid_features.float().to(self.device),
            'high': high_features.float().to(self.device),
            'highest': highest_features.float().to(self.device),
        }

        low_feat = self.low_feature_conv(features_dict['low'])
        low_feat = F.interpolate(low_feat, size=self.mid_feature_size, mode='bilinear', align_corners=False)

        mid_feat = self.mid_feature_conv(features_dict['mid'])
        mid_feat = torch.cat([low_feat, mid_feat], dim=1)
        mid_feat = self.mid_feature_mix_conv(mid_feat, y=None)
        mid_feat = F.interpolate(mid_feat, size=self.high_feature_size, mode='bilinear', align_corners=False)

        high_feat = self.high_feature_conv(features_dict['high'])
        high_feat = torch.cat([mid_feat, high_feat], dim=1)
        high_feat = self.high_feature_mix_conv(high_feat, y=None)

        highest_feat = self.highest_feature_conv(features_dict['highest'])

        # print('low_feat',low_feat.shape)
        # print('mid_feat',mid_feat.shape)
        # print('high_feat',high_feat.shape)
        # print('highest_feat',highest_feat.shape)
        highest_feat = torch.cat([high_feat, highest_feat], dim=1)
        highest_feat = self.highest_feature_mix_conv(highest_feat, y=None)
        highest_feat = F.interpolate(highest_feat, size=self.final_high_feature_size, mode='bilinear',
                                     align_corners=False)

        low_feat = F.interpolate(low_feat, size=20, mode='bilinear', align_corners=False)
        mid_feat = F.interpolate(mid_feat, size=40, mode='bilinear', align_corners=False)
        high_feat = F.interpolate(high_feat, size=80, mode='bilinear', align_corners=False)

        return [low_feat, mid_feat, high_feat], highest_feat
