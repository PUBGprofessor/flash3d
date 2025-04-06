# import pickle
# import gzip
# from pathlib import Path


# def load_seq_data(data_path, split):
#     file_path = data_path / f"{split}.pickle.gz"
#     with gzip.open(file_path, "rb") as f:
#         seq_data = pickle.load(f)
#     return seq_data

# path = Path(r"F:\3DGS_code\dataset\RealEstate10K")
# seq  = load_seq_data(path, "test")
# # print(seq['ffd8aa5c07187add']['poses'])
# print(seq.keys())

#####################################

# import torch

# n = torch.randn(1, 4, 5, 5, requires_grad=True)
# l = n[:, 0, :, :].unsqueeze(1)
# mask = torch.ones_like(n)
# mask[:, [1, 2], :, :] = 0
# rotation = l * mask
# m = rotation * 2
# sum = m.sum()
# sum.backward()
# print(n.grad)


##################################################
# import torch
# from PIL import Image
# import torchvision.transforms as T
# import matplotlib.pyplot as plt

# # 加载示例图像
# img = Image.open(r"F:\3DGS_code\dataset\DIV2K_valid_LR_bicubic\X2\0801x2.png")

# # model = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True, yoso_version='yoso-normal-v0-3')
# model = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)

# model.eval()
# normal_image =model(img)

# # Save or display the result
# normal_image.show("output/normal_map.png")

######################################################
# import torch
# from transformers import DPTImageProcessor, DPTForDepthEstimation
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np

# # ✅ 加载图像
# image = Image.open(r"F:\3DGS_code\dataset\DIV2K_valid_LR_bicubic\X2\0801x2.png").convert("RGB")

# # ✅ 加载处理器 & 模型
# processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas-normals")
# model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas-normals")
# model.eval()

# # ✅ 输入处理
# inputs = processor(images=image, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)

# # ✅ 输出包括：深度图、法向图
# predicted_depth = outputs.predicted_depth[0].cpu().numpy()
# predicted_normals = outputs.predicted_normals[0].cpu().numpy()  # shape: [3, H, W]

# # ✅ 可视化深度图
# depth_rescaled = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
# plt.imshow(depth_rescaled, cmap="inferno")
# plt.title("Estimated Depth")
# plt.axis("off")
# plt.show()

# # ✅ 可视化法向图（作为 RGB 显示）
# normals_rgb = (predicted_normals + 1) / 2  # [-1, 1] -> [0, 1]
# normals_rgb = np.transpose(normals_rgb, (1, 2, 0))
# plt.imshow(normals_rgb)
# plt.title("Estimated Normals")
# plt.axis("off")
# plt.show()

##################################
# 将单位法向量转换成四元数
# import numpy as np
 
# def unit_vector_to_quaternion(unit_vector):
#     w = np.sqrt(1 + unit_vector[0] + unit_vector[1] + unit_vector[2]) / 2
#     x = (unit_vector[2] - unit_vector[1]) / (4 * w)
#     y = (unit_vector[0] - unit_vector[2]) / (4 * w)
#     z = (unit_vector[1] - unit_vector[0]) / (4 * w)
#     return np.array([w, x, y, z])
 
# # Example unit vector (replace with your own values)
# unit_vector = np.array([ 0.01,  0.01, -1])
# # unit_vector = np.array([ 0.514385,  -0.0318042, 0.856969])
 
# # Convert unit vector to quaternion
# quaternion = unit_vector_to_quaternion(unit_vector)
 
# print("Quaternion:", quaternion)

####################################
import torch
import torch.nn.functional as F

# 假设类中方法写在这里（直接拷贝进来）
class DummyController:
    def compute_quaternion_from_normals(self, normals):
        """
        normals: (B, 3, H, W)
        return: (B, 4, H, W)
        """
        v1 = torch.tensor([0.0, 0.0, -1.0], device=normals.device).view(1, 3, 1, 1)  # reference vector
        v2 = F.normalize(normals, dim=1)  # ensure unit normal

        # 点积（用于计算角度）
        dot = (v1 * v2).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)  # (B, 1, H, W)
        theta = torch.acos(dot)  # (B, 1, H, W)
        half_theta = theta / 2

        # 特殊处理：反方向情况
        mask_opposite = (dot < -0.999).float()
        fallback_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=normals.device).view(1, 4, 1, 1)

        # 轴（v1 × v2）
        axis = torch.cross(v1.expand_as(v2), v2, dim=1)
        axis = F.normalize(axis + 1e-8, dim=1)  # 避免除以 0

        # 构建四元数
        w = torch.cos(half_theta)
        xyz = axis * torch.sin(half_theta)
        quat = torch.cat([w, xyz], dim=1)  # (B, 4, H, W)

        # 如果是反方向，使用 fallback 四元数
        quat = quat * (1 - mask_opposite) + fallback_quat * mask_opposite

        return F.normalize(quat, dim=1)

    def control_normal(self, outputs, gaussians_per_pixel=2):
        """
        将 normal 向量指定方向，转换为四元数，并更新 gauss_rotation 中每个 batch 第一个高斯的旋转。
        """
        normals = outputs[("normal", 0)]  # (B, 3, H, W)
        B, _, H, W = normals.shape
        new_quats = self.compute_quaternion_from_normals(normals)  # (B, 4, H, W)

        rot = outputs["gauss_rotation"]  # (B * gaussians_per_pixel, 4, H, W)

        # 构建 mask：选中第一个高斯
        mask = torch.zeros_like(rot[:, :1])  # (B * G, 1, H, W)
        for b in range(B):
            mask[b * gaussians_per_pixel] = 1.0  # 仅第一个高斯为1

        # 扩展 new_quats 到和 rot 相同 shape，用于 mask 替换
        expanded_quat = torch.zeros_like(rot)
        for b in range(B):
            expanded_quat[b * gaussians_per_pixel] = new_quats[b]

        # mask 替换：mask==1 的位置取 expanded_quat，其余保持原值
        updated_rot = rot * (1.0 - mask) + expanded_quat * mask

        outputs["gauss_rotation"] = updated_rot


B, G, H, W = 2, 2, 4, 4  # batch size 2，每个像素2个高斯
controller = DummyController()

normals = torch.zeros((B, 3, H, W), dtype=torch.float32)
# 随机生成两个法向量（单位向量）
normal0 = F.normalize(torch.randn(3, 1, 1), dim=0)  # shape (3, 1, 1)
normal1 = F.normalize(torch.randn(3, 1, 1), dim=0)

# 应用于 batch 中的 normals
normals[0] = normal0
normals[1] = normal1


# 构造 gauss_rotation 为全1（便于看出是否替换）
gauss_rotation = torch.ones((B * G, 4, H, W))

# 构造 outputs dict
outputs = {
    ("normal", 0): normals,
    "gauss_rotation": gauss_rotation.clone()
}

# 保存原高斯旋转用于比较
original_rot = outputs["gauss_rotation"].clone()

# 执行替换
controller.control_normal(outputs, gaussians_per_pixel=2)

# 验证第一个被替换，其它未变
print("原始第0个高斯旋转：", original_rot[0, :, 0, 0])
print("更新后第0个高斯旋转：", outputs["gauss_rotation"][0, :, 0, 0])
print("是否不同：", not torch.allclose(original_rot[0], outputs["gauss_rotation"][0]))

print("\n原始第1个高斯旋转：", original_rot[1, :, 0, 0])
print("更新后第1个高斯旋转：", outputs["gauss_rotation"][1, :, 0, 0])
print("是否相同：", torch.allclose(original_rot[1], outputs["gauss_rotation"][1]))
