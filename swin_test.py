import torch
import torch.nn as nn
from models.encoder.resnet_encoder import ResnetEncoder
from models.encoder.swin_encoder import SwinEncoder


encoder = ResnetEncoder(
            num_layers=50,
            pretrained=True,
            bn_order="pre_bn",
        )
encoder.encoder.conv1 = nn.Conv2d(
                    # 4,
                    7,
                    encoder.encoder.conv1.out_channels,
                    kernel_size = encoder.encoder.conv1.kernel_size,
                    padding = encoder.encoder.conv1.padding,
                    stride = encoder.encoder.conv1.stride
                )
encoder = SwinEncoder(pretrained=True, bn_order="pre_bn", desired_height=320, desired_width=448)

old_proj = encoder.patch_embed.proj
encoder.patch_embed.proj = torch.nn.Conv2d(
    in_channels=7,
    out_channels=old_proj.out_channels,
    kernel_size=old_proj.kernel_size,
    stride=old_proj.stride,
    padding=old_proj.padding,
    bias=old_proj.bias is not None
)
x = torch.randn(2, 7, 320, 448)  # Example input tensor
# x = torch.randn(2, 3, 224, 224)  # Example input tensor
features = encoder(x)
print("Output features shape:", [f.shape for f in features])  # Should print shapes of feature maps
# [torch.Size([2, 64, 160, 224]), torch.Size([2, 256, 80, 112]), torch.Size([2, 512, 40, 56]), torch.Size([2, 1024, 20, 28]), torch.Size([2, 2048, 10, 14])]
# [torch.Size([2, 32, 320, 448]), torch.Size([2, 128, 80, 112]), torch.Size([2, 256, 40, 56]), torch.Size([2, 512, 20, 28]), torch.Size([2, 1024, 10, 14])]