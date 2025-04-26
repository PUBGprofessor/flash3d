import torch

model = torch.hub.load(
            "lpiccinelli-eth/UniDepth", "UniDepth", #version=cfg.model.depth.version, 
            #backbone=cfg.model.depth.backbone, 
            pretrained=True, trust_repo=True, 
            force_reload=True
        )