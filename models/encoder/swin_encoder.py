import numpy as np

import torch
import torch.nn as nn
from timm.models.swin_transformer import swin_base_patch4_window7_224, SwinTransformer

class SwinEncoder(nn.Module):
    """Pytorch module for a Swin Transformer encoder, adapted to match ResnetEncoder interface."""
    def __init__(self, pretrained=True, bn_order=None, img_size=224):
        super(SwinEncoder, self).__init__()

        self.bn_order = bn_order
        self.model: SwinTransformer = swin_base_patch4_window7_224(pretrained=pretrained)

        self.num_ch_enc = np.array([128, 256, 512, 1024])  # For swin_base

    def forward(self, input_image):
        x = (input_image - 0.45) / 0.225

        features = []

        # timm SwinTransformer returns features if forward_features is called
        # expects (B, C, H, W), output is list of feature maps from 4 stages
        x = self.model.forward_features(x, return_all_features=True)

        # return_all_features=True gives [stage1, stage2, stage3, stage4]
        features.extend(x)  # Each with shape (B, C, H/4, W/4), (B, C, H/8, W/8), etc.

        return features
