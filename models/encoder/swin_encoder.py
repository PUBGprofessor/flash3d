import numpy as np
from timm import create_model
import torch
import torch.nn as nn
from timm.models.swin_transformer import swin_base_patch4_window7_224, SwinTransformer

class SwinEncoder(nn.Module):
    """Pytorch module for a Swin Transformer encoder, adapted to match ResnetEncoder interface."""
    def __init__(self, pretrained=True, bn_order=None, desired_height=256, desired_width=384):
        super(SwinEncoder, self).__init__()

        self.bn_order = bn_order
        # self.model: SwinTransformer = swin_base_patch4_window7_224(pretrained=pretrained)
        self.model: SwinTransformer = create_model(
                'swin_base_patch4_window7_224',
                pretrained=True,
                img_size=(desired_height, desired_width),  # 例如 (256, 256) 或 (240, 320)
                features_only=False  # 设置为 True 会返回中间层特征
            )

        self.cov = nn.Conv2d(
            in_channels=7,  # 输入通道数，假设输入是 RGB + 深度 + 法线图
            out_channels=32,  # 输出通道数，假设输出是 RGB 图像
            kernel_size=1,  # 卷积核大小为 1x1
            stride=1,  # 步幅为 1
            padding=0,  # 无填充
        )

        # self.num_ch_enc = np.array([128, 128, 256, 512, 1024])  # For swin_base
        self.num_ch_enc = np.array([32, 128, 256, 512, 1024])  # 改第一层后

         # 提前解包模型关键组件
        self.patch_embed = self.model.patch_embed
        self.layers = self.model.layers  # 4 个 stage
        self.norm = self.model.norm if hasattr(self.model, 'norm') else nn.Identity()

    def forward(self, x):
        # features = []

        # patch embedding
        features = []
        features.append(nn.functional.avg_pool2d(self.cov(x), kernel_size=2, stride=2))  # 添加 cov 层的输出作为 stage0

        x = self.patch_embed(x)
        # features.append(x)  # 把 patch embed 也当作 stage0
        for i in range(4):
            x = self.layers[i](x)
            features.append(x)

        return [features[0]] + [x.permute(0, 3, 1, 2) for x in features[1:]]
