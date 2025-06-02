import torch

# model = torch.hub.load(
#             "lpiccinelli-eth/UniDepth", "UniDepth", #version=cfg.model.depth.version, 
#             #backbone=cfg.model.depth.backbone, 
#             pretrained=True, trust_repo=True, 
#             force_reload=True
#         )

model = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)
# model = torch.hub.load(
#             "Stable-X/StableNormal", 
#             "StableNormal_turbo", 
#             trust_repo=True, 
#             yoso_version='yoso-normal-v0-3'
#         )