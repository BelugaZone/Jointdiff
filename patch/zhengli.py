# declare the cross atten net
self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

####################需要的函数##########


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor, temperature=None):
        not_mask = torch.ones(x.shape[0], x.shape[-2], x.shape[-1], dtype=torch.bool, device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        if temperature is None:
            temperature = self.temperature
        dim_t = temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def get_reference_points(
        offset=None, 
        num_token:int=196, 
        train_dense:int=2, 
        dense_sample:int=2, 
        h_sample:int=4,
        z_range=[1, 50],
        x_range=[-25, 25],
        y_range=[0, 3],
    ):
        # train_dense is query number
        # dense_sample dense_sample is the reference point number of each query: num_ref = dense_sample ** 2 * y_sample_num
        H, W = int(num_token ** 0.5), int(num_token ** 0.5)
        assert H * W == num_token
        H = (H - 1) * train_dense + 1
        W = (W - 1) * train_dense + 1
        if offset is not None:
            # assert NotImplementedError
            if offset[0] > 0:
                H = H - 1
                W = W - 1
                x_range[0] = x_range[0] + offset[1].item()
                z_range[0] = z_range[0] + offset[0].item()

        H = H * dense_sample
        W = W * dense_sample
        Z = h_sample
        
        # HWZ are in BEV space, Z is height
        w = torch.linspace(0.5, W - 0.5, W).view(1, 1, W).expand(Z, H, W) / W # cam_x 
        h = torch.linspace(0.5, H - 0.5, H).view(1, H, 1).expand(Z, H, W) / H # cam_z 
        z = torch.linspace(0.5, Z - 0.5, Z).view(Z, 1, 1).expand(Z, H, W) / Z # cam_y
        
        ref_3d = torch.stack((w, h, z), -1) # (Z, W, H, 3) 
        ref_3d = ref_3d.reshape(Z, W//dense_sample, dense_sample, H//dense_sample, dense_sample, 3) 
        ref_3d = ref_3d.permute(0, 2, 4, 1, 3, 5).reshape(Z * dense_sample * dense_sample, -1, 3) # (h_samples * dense * dense, num_token, 3))
        # return ref_3d[None] # (1, h_samples * dense * dense, num_token, 3) 
        ref_3d[..., 0] = ref_3d[..., 0] * (x_range[1] - x_range[0]) + x_range[0]
        ref_3d[..., 1] = ref_3d[..., 1] * (z_range[1] - z_range[0]) + z_range[0]
        ref_3d[..., 2] = ref_3d[..., 2] * (y_range[1] - y_range[0]) + y_range[0]
        
        return ref_3d[None] # (1, h_samples * dense * dense, num_token, 3) in camera

def get_reference_points(H, W, Z=8, num_points_in_pillar=4,bs=1, device='cuda', dtype=torch.float):
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
        这些3d点的获取方法比较简单：
            对BEV空间（HW为像素分辨率）上进行采样，
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
        

    def ms_features_preprocess(self, norm_hidden_states):
            # 将单尺度的norm_hidden_states转换为兼容格式
        if len(norm_hidden_states.shape) == 3:
            # 序列形式 [bs, seq_len, c]
           bs, seq_len, c = norm_hidden_states.shape
           h = int(seq_len ** 0.5)
           w = h
           # 重塑为空间特征图 [bs, c, h, w]
           src = norm_hidden_states.permute(0, 2, 1).contiguous()
           src = src.reshape(bs, c, h, w)
        elif len(norm_hidden_states.shape) == 4:
            # 空间形式 [bs, c, h, w]
           src = norm_hidden_states
           bs, c, h, w = src.shape
        else:
           raise ValueError(f"Unexpected input shape: {norm_hidden_states.shape}")

         # 创建单尺度特征字典（即使只有一个层级也需字典格式）
        features = {'1': src}
    
         # 以下保持原有逻辑但适配单尺度
        src_flatten = []
        spatial_shapes = []
        pos_embed_flatten = []
    
        for k, src_layer in features.items():
            bs, c, h, w = src_layer.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
        
        # 位置编码处理
            pos_embed = self.pos_emb(src_layer)  # 使用自己的位置编码器
            pos_embed_flatten.append(pos_embed.flatten(2).transpose(1, 2))
        
        # 特征展平
            src_flat = src_layer.flatten(2).transpose(1, 2)  # [bs, c, h, w] → [bs, h*w, c]
            src_flatten.append(src_flat)
    
        # 单尺度不需要拼接
        src_flatten = src_flatten[0]  # [bs, seq_len, c]
        pos_embed_flatten = pos_embed_flatten[0]  # [bs, seq_len, c]
    
         # 空间形状信息
        spatial_shapes = torch.as_tensor(
            [spatial_shapes[0]],  # 保持为列表形式
            dtype=torch.long, 
            device=src_flatten.device
         )  # 形状: [1, 2]
    
        # 单层级起始索引始终为0
        level_start_index = torch.zeros(1, dtype=torch.long, device=src_flatten.device)
    
        return src_flatten, spatial_shapes, level_start_index, pos_embed_flatten

    
    def cam2img(self, sampling_locations_cam, instrinsic, img_shape):
        # sampling_location_cam.shape b, n_sample, n_patch, 3
        crop_height = img_shape[0]
        crop_width = img_shape[1]
        
        h, w = img_shape
        f, cu, cv = instrinsic[:, 0, 0], instrinsic[:, 0, 2], instrinsic[:, 1, 2]
        x, z, y = sampling_locations_cam[..., 0], sampling_locations_cam[..., 1], sampling_locations_cam[..., 2]
        f = f[:, None, None]
        cu = cu[:, None, None]
        cv = cv[:, None, None]
        z = torch.clamp(z, min=0.1)
        v = (y * f) / z + cv  # [b, num_patch] height
        u = (x * f) / z + cu  # [b, num_patch] width
        sampling_locations = torch.stack([u, v], dim=-1) # [b, num_patch, 2]

        # sampling_locations = sampling_locations / torch.tensor([crop_width, crop_height], dtype=torch.float32, device=instrinsic.device)
        sampling_locations[..., 0] = sampling_locations[..., 0] / crop_width
        sampling_locations[..., 1] = sampling_locations[..., 1] / crop_height
        return sampling_locations
    
    def get_valid(self, ref):
        # ref.shape b, n_sample, n_patch, 2
        valid = (ref[..., 0] > 0) & \
                (ref[..., 0] < 1) & \
                (ref[..., 1] > 0) & \
                (ref[..., 1] < 1)
                
        valid_union = valid.any(dim=0).any(dim=0)
        valid = valid.any(dim=1)
        ref = ref[..., valid_union, :]
        return ref, valid_union, valid

    def get_attn_mask(method:str="inner_local", num_token=625, kernel_size=5):
        if method == 'inner_local':
            H, W = int(num_token ** 0.5), int(num_token ** 0.5)
            assert H * W == num_token
            assert kernel_size % 2 == 1
            attn_mask = torch.zeros(num_token, num_token)
            indices = torch.arange(num_token)
            attn_mask[indices, indices] = 1
            attn_mask = attn_mask.view(-1, H, W)
            max_pool2d = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size//2)
            attn_mask = max_pool2d(attn_mask)
            attn_mask[attn_mask < 1] = float('-inf')
            attn_mask[attn_mask == 1] = 0
            return attn_mask.reshape(num_token, num_token)
        else:
            raise NotImplementedError

    def get_query_pos_embed(self, temperature=20):
        H, W = int(self.num_token ** 0.5), int(self.num_token ** 0.5)
        pos = self.query_pos_emb(torch.zeros(1, self.dim_model, H, W), temperature=temperature)
        return pos.flatten(2).permute(0, 2, 1)
     
  
    
 
    
 ##################################################################3       


def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

###### main forward
tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask,src_valid_ratios)
tgt = tgt + self.dropout1(tgt2)
tgt = self.norm1(tgt)


######### 各参数的获取过程

features, spatial_shapes, level_start_index, features_pos = self.ms_features_preprocess(ms_features)


##### query_pos     reference_points    src_padding_mask,s  rc_valid_ratios
ref_cam = self.get_reference_points(
            offset=offset, 
            num_token=self.num_patches, 
            # train_dense=2 if self.training else 1, 
            train_dense=1,
            dense_sample=self.cfg.sample_dense,
            h_sample=self.cfg.y_sample_num,
            z_range=[1, 50],
            x_range=[-25, 25],
            y_range=[0, 3],
            # y_range = [1., 2.5] # argoverse using
        )

ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype)


b = features.shape[0]
ref_cam = ref_cam.repeat(b, 1, 1, 1).to(device=features.device)
        
ref_img = self.cam2img(ref_cam, calib, img_shape=[256, 256])
       
        
reference_select, valid_union, valid = self.get_valid(ref_img)
        
self.query_pos_emb = PositionEmbeddingSine(num_pos_feats=dim_model//2, normalize=True)

query_pos = self.get_query_pos_embed().to(query.device)

self.attn_mask = self.get_attn_mask(method='inner_local', num_token=num_token, kernel_size=mask_kernel_size)










# self.with_pos_embed(tgt, query_pos)    查询向量
# reference_points     参考点
# src    值向量KV
# src_spatial_shapes  每个特征层级的空间尺寸(H,W)
# level_start_index   (层级起始索引)​
# src_padding_mask  填充掩码
# src_valid_ratios  有效比例  为none


