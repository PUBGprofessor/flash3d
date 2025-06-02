import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from models.encoder.resnet_encoder import ResnetEncoder
from models.decoder.resnet_decoder import ResnetDecoder, ResnetDepthDecoder

from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
to_tensor = ToTensor()

class UniDepthExtended(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # 深度预测模型
        # 这里使用了lpiccinelli-eth/UniDepth的预训练模型
        # 调用方法为self.unidepth.infer(inputs["color_aug", 0, 0], intrinsics=inputs[("K_src", 0)])
        self.unidepth = torch.hub.load(
            "lpiccinelli-eth/UniDepth", "UniDepth", version=cfg.model.depth.version, 
            backbone=cfg.model.depth.backbone, pretrained=True, trust_repo=True, 
            force_reload=True
        )

        self.StableNormal = torch.hub.load(
            "Stable-X/StableNormal", 
            "StableNormal_turbo", 
            trust_repo=True, 
            yoso_version='yoso-normal-v0-3'
        )

        self.parameters_to_train = []
        if cfg.model.backbone.name == "resnet":
            # ResnetEncoder
            self.encoder = ResnetEncoder(
                num_layers=cfg.model.backbone.num_layers,
                pretrained=cfg.model.backbone.weights_init == "pretrained",
                bn_order=cfg.model.backbone.resnet_bn_order,
            )
            # change encoder to take depth as conditioning
            # 因为输出的通道数是4，所以需要将输入的通道数改为4
            if cfg.model.backbone.depth_cond:
                self.encoder.encoder.conv1 = nn.Conv2d(
                    # 4,
                    7,
                    self.encoder.encoder.conv1.out_channels,
                    kernel_size = self.encoder.encoder.conv1.kernel_size,
                    padding = self.encoder.encoder.conv1.padding,
                    stride = self.encoder.encoder.conv1.stride
                )
            self.parameters_to_train += [{"params": self.encoder.parameters()}]
            models = {}
            if cfg.model.gaussians_per_pixel > 1:
                # 如果需要预测后面的高斯分布，则需要使用ResnetDepthDecoder（一个DepthDecoder预测全部）
                models["depth"] = ResnetDepthDecoder(cfg=cfg, num_ch_enc=self.encoder.num_ch_enc)
                self.parameters_to_train +=[{"params": models["depth"].parameters()}]
            for i in range(cfg.model.gaussians_per_pixel):
                # 预测其他高斯参数
                models["gauss_decoder_"+str(i)] = ResnetDecoder(cfg=cfg,num_ch_enc=self.encoder.num_ch_enc)
                self.parameters_to_train += [{"params": models["gauss_decoder_"+str(i)].parameters()}]
                if cfg.model.one_gauss_decoder:
                    break

            # 此时self.models里面有。。。
            self.models = nn.ModuleDict(models)

    def get_parameter_groups(self):
        # only the resnet encoder and gaussian parameter decoder are optimisable
        return self.parameters_to_train
    
    def forward(self, inputs):

        # prediting the depth for the first layer with pre-trained depth
        # 如果已有的深度图，则直接使用，没有则使用unidepth预测
        if ('unidepth', 0, 0) in inputs.keys() and inputs[('unidepth', 0, 0)] is not None:
            depth_outs = dict()
            depth_outs["depth"] = inputs[('unidepth', 0, 0)]
        else:
            with torch.no_grad():
                intrinsics = inputs[("K_src", 0)] if ("K_src", 0) in inputs.keys() else None # 内参矩阵
                depth_outs = self.unidepth.infer(inputs["color_aug", 0, 0], intrinsics=intrinsics)

        # 这里的depth_outs["depth"]是第一层高斯的深度，形状为(B, 1, H, W)

        # 如果已有的法向图，则直接使用，没有则使用StableNormal预测
        if ('normal', 0, 0) in inputs.keys() and inputs[('normal', 0, 0)] is not None:
            depth_outs = dict()
            depth_outs["normal"] = inputs[('normal', 0, 0)]
        else:
            with torch.no_grad():
                depth_outs["normal"] = []
                for i in range(inputs["color_aug", 0, 0].shape[0]):
                    img_tensor = inputs["color_aug", 0, 0][i]  # [C, H, W]
                    img_pil = to_pil_image(img_tensor)
                    normal_pil = self.StableNormal(img_pil)  # 结果是 PIL.Image
                    normal_tensor = to_tensor(normal_pil)  # 转为 Tensor，shape: [3, H, W]
                    depth_outs["normal"].append(normal_tensor)
                depth_outs["normal"] = torch.stack(depth_outs["normal"]).to("cuda")  # [B, 3, H, W]

        outputs_gauss = {}

        outputs_gauss[("K_src", 0)] = inputs[("K_src", 0)] if ("K_src", 0) in inputs.keys() else depth_outs["intrinsics"]
        outputs_gauss[("inv_K_src", 0)] = torch.linalg.inv(outputs_gauss[("K_src", 0)])  # 逆内参矩阵

        if self.cfg.model.backbone.depth_cond:
            # division by 20 is to put depth in a similar range to RGB
            input = torch.cat([inputs["color_aug", 0, 0], depth_outs["depth"] / 20.0], dim=1) # 拼接深度图和RGB图（共4 * H * W）
        else:
            input = inputs["color_aug", 0, 0]

        input = torch.cat([input, depth_outs["normal"]], dim=1) # 拼接法向图和RGB图（共7 * H * W）
        # input = torch.cat([input, (depth_outs["normal"] + 1) * 100], dim=1) # 拼接法向图和RGB图（共7 * H * W）
        # encode the input image
        encoded_features = self.encoder(input)

        # predict multiple gaussian depths
        if self.cfg.model.gaussians_per_pixel > 1:
            depth = self.models["depth"](encoded_features) # depth[("depth", 0)]： (B * (gaussians_per_pixel - 1), 1, H, W)
            """
            epth[("depth", 0)] 初始形状是 (B*N, ...)，其中：
            B 是批量大小（batch size）。
            N = gaussians_per_pixel - 1，表示模型预测的是 额外的 N 个高斯深度增量（而不是绝对深度）。
            rearrange(..., "(b n) ... -> b n ...", n=...) 作用：
            把 (B*N, ...) 变回 (B, N, ...)，这样每个像素点上就有 N 个增量深度值。
            """
            depth[("depth", 0)] = rearrange(depth[("depth", 0)], "(b n) ... -> b n ...", n=self.cfg.model.gaussians_per_pixel - 1)
            depth[("depth", 0)] = torch.cumsum(torch.cat((depth_outs["depth"][:,None,...], depth[("depth", 0)]), dim=1), dim=1)  # 拼接后(B, gaussians_per_pixel, 1, H, W)
            outputs_gauss[("depth", 0)] = rearrange(depth[("depth", 0)], "b n c ... -> (b n) c ...", n = self.cfg.model.gaussians_per_pixel) # (B * gaussians_per_pixel, 1, H, W)
        else:
            outputs_gauss[("depth", 0)] = depth_outs["depth"]
        
        outputs_gauss[("normal", 0)] = depth_outs["normal"]
        # predict multiple gaussian parameters
        gauss_outs = dict()
        for i in range(self.cfg.model.gaussians_per_pixel):
            outs = self.models["gauss_decoder_"+str(i)](encoded_features) # 仅输入由图像和初始深度图拼接而成的特征，够吗？感觉可以加上预测出来的本层的深度图
            if self.cfg.model.one_gauss_decoder:
                gauss_outs |= outs
                break
            else:
                for key, v in outs.items():
                    gauss_outs[key] = outs[key] if i==0 else torch.cat([gauss_outs[key], outs[key]], dim=1)
                    
        for key, v in gauss_outs.items():
            gauss_outs[key] = rearrange(gauss_outs[key], 'b n ... -> (b n) ...')
        outputs_gauss |= gauss_outs

        return outputs_gauss