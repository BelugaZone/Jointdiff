from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch

from deformable_detr.position_encoding import PositionEmbeddingSine





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




@force_fp32(apply_to=('reference_points', 'img_metas'))
def point_sampling(self, reference_points, pc_range,  img_metas):

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4) # 坐标变换矩阵，将3d点映射到2d图像平面上去 #[1, 6, 4, 4]
        reference_points = reference_points.clone() # [1, 4, 22500, 3]
        # 映射到真实世界的3d坐标
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]
        # 构建齐次坐标系
        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1) # [1, 4, 22500, 4]
        # 变换 [4, 1, 22500,4]
        reference_points = reference_points.permute(1, 0, 2, 3)

        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1) # [4, 1, 6, 22500, 4, 1]

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1) # [4, 1, 6, 22500, 4, 4]

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1) # 计算处每个参考点在6个view的坐标 [4, 1, 6, 22500, 4], 最后一维是坐标的齐次值
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps) # 坐标系转换？x，y坐标/z 将从齐次坐标系转换到笛卡尔直角坐标系，这时候还是像素坐标呢？

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1] # 这里的物理意义是什么？归一化了
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0] #[4, 1, 6, 22500, 2]现在的维度

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0)) # 归一化之后将越界的点过滤
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask) # 处理那些不可计算的部分，如果为nan，就补0
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4) # [6, 1, 22500, 4, 2]
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask















        


ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype)

reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs['img_metas'])



###encoder中最外层调用

 output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs)



###layer中调用

identity = query

query = self.attentions[attn_index](
                    query,                                      #bev查询
                    key,                                        #特征图
                    value,                                      #特征图
                    identity if self.pre_norm else None,
                    query_pos=query_pos,                        #bev位置编码
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,              # 多尺度特征图尺寸
                    level_start_index=level_start_index,
                    **kwargs)




queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)



output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)





bev_queries = self.bev_embedding.weight.to(dtype)

bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
bev_pos = self.positional_encoding(bev_mask).to(dtype)


##### 图像特征的编码，作为k和v，只添加了相机标识和层级标识的编码

   feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)



MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
 # apply会自动执行forward方法
output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)

