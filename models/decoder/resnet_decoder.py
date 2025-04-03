import torch
import numpy as np
from einops import rearrange
from collections import OrderedDict

from models.encoder.layers import *
from models.decoder.gaussian_decoder import get_splits_and_inits, GaussianDecoder


class ResnetDecoder(nn.Module):
    """Pytorch module for a resnet decoder"""
    def __init__(self, cfg, num_ch_enc, use_skips=True):
        super().__init__()

        self.cfg = cfg
        self.use_skips = use_skips # use_skips=True 说明 解码器使用跳跃连接（skip connections），即在上采样过程中，除了当前层的特征外，还会 融合编码器中对应层的特征。
        self.num_ch_enc = num_ch_enc # np.array([64, 64, 128, 512, 2048])输出的featuresk列表每层的通道数
        self.num_ch_dec = np.array(cfg.model.backbone.num_ch_dec) # resnet.yaml: [32,32,64,128,256]

        self.split_dimensions, scales, biases = get_splits_and_inits(cfg)
        self.num_output_channels = sum(self.split_dimensions)

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1): # [4, 3, 2, 1, 0]
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1] # [2048, 256, 128, 64, 32]
            num_ch_out = self.num_ch_dec[i] # [256, 128, 64, 32, 32]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i] # [256, 128, 64, 32, 32]
            if self.use_skips and i > 0: # use_skips=True 说明 解码器使用跳跃连接（skip connections），即在上采样过程中，除了当前层的特征外，还会 融合编码器中对应层的特征。
                num_ch_in += self.num_ch_enc[i - 1] # [256 + 512, 128 + 128, 64 + 64, 32 + 64, 32]
            num_ch_out = self.num_ch_dec[i] # [256, 128, 64, 32, 32]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.out = nn.Conv2d(self.num_ch_dec[0], self.num_output_channels, 1) # 32 -> 3+1+3+4+3+9=23

        # gaussian parameters initialisation
        start_channel = 0
        for out_channel, scale, bias in zip(self.split_dimensions, scales, biases):
            nn.init.xavier_uniform_(
                self.out.weight[start_channel:start_channel+out_channel,
                                :, :, :], scale)
            nn.init.constant_(
                self.out.bias[start_channel:start_channel+out_channel], bias)
            start_channel += out_channel

        # gaussian parameters activation
        self.gaussian_decoder = GaussianDecoder(cfg)

    def forward(self, input_features):
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x, mode=self.cfg.model.backbone.upsample_mode)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, dim=1)
            x = self.convs[("upconv", i, 1)](x)
        
        x = self.out(x)
        out = self.gaussian_decoder(x, self.split_dimensions)

        return out


class ResnetDepthDecoder(nn.Module):
    """Pytorch module for a resnet depth decoder"""
    def __init__(self, cfg, num_ch_enc, use_skips=True):
        super().__init__()
        
        self.cfg = cfg
        self.scales = cfg.model.scales
        self.use_skips = use_skips
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # 如果没有深度信息，那么需要预测所有的深度，若已有unidepth深度信息，则只需要预测除第一层外的深度
        self.num_output_channels = cfg.model.gaussians_per_pixel - 1 if "unidepth" in cfg.model.name else cfg.model.gaussians_per_pixel

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            out = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            self.convs[("outconv", s)] = out
            nn.init.xavier_uniform_(out.conv.weight, cfg.model.depth_scale)
            nn.init.constant_(out.conv.bias, cfg.model.depth_bias)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        
        # depth activation function
        if cfg.model.depth_type in ["disp", "disp_inc"]:
            self.activate = nn.Sigmoid()
        elif cfg.model.depth_type == "depth":
            self.activate = nn.Softplus()
        elif cfg.model.depth_type == "depth_inc":
            self.activate = torch.exp # 此处使用这个

    def forward(self, input_features):
        outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x, mode=self.cfg.model.backbone.upsample_mode)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, dim=1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:         
                output = self.convs[("outconv", i)](x)
                if self.cfg.model.depth_type == "depth_inc":
                    output = torch.clamp(output, min=-10.0, max=6.0)
                output = rearrange(self.activate(output), "b (n c) ... -> (b n) c ...", n=self.num_output_channels)
                if self.cfg.model.depth_type in ["disp", "disp_inc"]:
                    output = disp_to_depth(output, self.cfg.model.min_depth, self.cfg.model.max_depth)
                outputs[("depth", i)] = output
        return outputs # outputs[("depth", i)] ：(B * (gaussians_per_pixel - 1), 1, H_i, W_i)
