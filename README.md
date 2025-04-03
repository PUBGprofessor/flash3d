[![arXiv](https://img.shields.io/badge/arXiv-2406.04343-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/abs/2406.04343)
[![ProjectPage](https://img.shields.io/badge/Project_Page-Flash3D-blue)](https://www.robots.ox.ac.uk/~vgg/research/flash3d/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Demo-yellow)](https://huggingface.co/spaces/szymanowiczs/flash3d) 


# Flash3D: Feed-Forward Generalisable 3D Scene Reconstruction from a Single Image


<p align="center">
  <img src="assets/teaser_video.gif" alt="animated" />
</p>

> [Flash3D: Feed-Forward Generalisable 3D Scene Reconstruction from a Single Image](https://www.robots.ox.ac.uk/~vgg/research/flash3d/)  
> Stanislaw Szymanowicz, Eldar Insafutdinov, Chuanxia Zheng, Dylan Campbell, João F. Henriques, Christian Rupprecht, Andrea Vedaldi  
> *[arXiv 2406.04343](https://arxiv.org/pdf/2406.04343.pdf)*  

# News
- [x] `19.07.2024`: Training code and data release

# Setup

## Create a python environment

Flash3D has been trained and tested with the followings software versions:

- Python 3.10
- Pytorch 2.2.2
- CUDA 11.8
- GCC 11.2 (or more recent)

Begin by installing CUDA 11.8 and adding the path containing the `nvcc` compiler to the `PATH` environmental variable.
Then the python environment can be created either via conda:

```sh
conda create -y python=3.10 -n flash3d
conda activate flash3d
```

or using Python's venv module (assuming you already have access to Python 3.10 on your system):

```sh
python3.10 -m venv .venv
. .venv/bin/activate
```

Finally, install the required packages as follows:

```sh
pip install -r requirements-torch.txt --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Download training data

### RealEstate10K dataset

For downloading the RealEstate10K dataset we base our instructions on the [Behind The Scenes](https://github.com/Brummi/BehindTheScenes/tree/main?tab=readme-ov-file#-datasets) scripts.
First you need to download the video sequence metadata including camera poses from https://google.github.io/realestate10k/download.html and unpack it into `data/` such that the folder layout is as follows:

```
data/RealEstate10K/train
data/RealEstate10K/test
```

Finally download the training and test sets of the dataset with the following commands:

```sh
python datasets/download_realestate10k.py -d data/RealEstate10K -o data/RealEstate10K -m train
python datasets/download_realestate10k.py -d data/RealEstate10K -o data/RealEstate10K -m test
```

This step will take several days to complete. Finally, download additional data for the RealEstate10K dataset.
In particular, **we provide pre-processed COLMAP cache containing sparse point clouds which are used to estimate the scaling factor for depth predictions.**
The last two commands filter the training and testing set from any missing video sequences.

```sh
sh datasets/dowload_realestate10k_colmap.sh
python -m datasets.preprocess_realestate10k -d data/RealEstate10K -s train
python -m datasets.preprocess_realestate10k -d data/RealEstate10K -s test
```

## Download and evaluate the pretrained model

We provide model weights that could be downloaded and evaluated on RealEstate10K test set:

```sh
python -m misc.download_pretrained_models -o exp/re10k_v2
sh evaluate.sh exp/re10k_v2
```

## Training

In order to train the model on RealEstate10K dataset execute this command:
```sh
python train.py \
  +experiment=layered_re10k \
  model.depth.version=v1 \
  train.logging=false 
```

For multiple GPU, we can run with this command:
```sh
sh train.sh
```
You can modify the cluster information in ```configs/hydra/cluster```.


## BibTeX
```
@article{szymanowicz2024flash3d,
      author = {Szymanowicz, Stanislaw and Insafutdinov, Eldar and Zheng, Chuanxia and Campbell, Dylan and Henriques, Joao and Rupprecht, Christian and Vedaldi, Andrea},
      title = {Flash3D: Feed-Forward Generalisable 3D Scene Reconstruction from a Single Image},
      journal = {arxiv},
      year = {2024},
}
```



流程：

1. 下载数据集及数据集格式：

`out_path` 目录结构如下：

```python
data/RealEstate10k/
├── train/  （训练集）
│   ├── scene_001/ # 这里的文件名为txt的文件名，而不是YouTube ID
│   │   ├── 123456789.jpg  （从时间戳 123456789ms 提取的帧）
│   │   ├── 123457890.jpg
│   │   ├── ...
│   ├── scene_002/
│   │   ├── 987654321.jpg
│   │   ├── ...
│   ├── ...
├── test/   （测试集）
│   ├── scene_101/
│   │   ├── 112233445.jpg
│   │   ├── ...
│   ├── ...
```

每个`scene_xxx`文件夹对应一个YouTube视频，文件夹内是该视频按时间戳截取的帧图像，分辨率为360p（256 * 384）。

，RealEstate10K数据集的Flash3D版本可能包含：

```python
data/RealEstate10K/
├── test.pickle.gz  （测试集索引）
├── train.pickle.gz  （训练集索引） # 格式：字典{“文件名（场景名）”：{'timestamps'：[1 时间戳], 'intrinsics':[6 内参], 'poses':[3 * 4 外参]， ...}
├── pcl.test.tar  （测试集的点云数据）
├── pcl.train.tar  （训练集的点云数据）
├── valid_seq_ids.train.pickle.gz  （有效训练序列）
├── SHA512SUMS  （哈希值校验文件）
```

解压 `.tar` 可能会得到点云文件（如 `.ply` 或 `.npy`），可以用于3D重建。



2. 输入格式（input）

### **`inputs` 的结构**

```python
{
    # 这里的frame_id为0, 1, 2
    ("frame_id", 0): str,                  # 当前帧的唯一标识符
    ("K_tgt", frame_name): torch.Tensor,   # 目标帧的相机内参矩阵 (3×3)
    ("K_src", frame_name): torch.Tensor,   # 源帧的相机内参矩阵 (3×3)
    ("inv_K_src", frame_name): torch.Tensor, # 源帧内参矩阵的逆 (3×3)
    ("color", frame_name, 0): torch.Tensor,  # 目标帧的 RGB 图像 (C×H×W)
    ("color_aug", frame_name, 0): torch.Tensor, # 经过数据增强的 RGB 图像 (C×H×W)
    ("T_c2w", frame_name): torch.Tensor,   # 当前帧的相机到世界坐标变换矩阵 (4×4)
    ("T_w2c", frame_name): torch.Tensor,   # 世界坐标到当前帧相机坐标变换矩阵 (4×4)
    ("unidepth", frame_name, 0): Optional[torch.Tensor], # 深度图 (1×H×W)
    ("depth_sparse", 0): Optional[torch.Tensor],  # 稀疏点云数据
    ("scale_colmap", 0): Optional[torch.Tensor]   # 估算的深度尺度（如果使用 RANSAC）
    # 上述为dataloader返回的，下面的是在流程中加上去的
    "target_frame_ids": cfg.model.gauss_novel_frames # [1, 2]
}
```

3. 模型结构

![image-20250402110258819](C:\Users\刘佳豪\AppData\Roaming\Typora\typora-user-images\image-20250402110258819.png)

（1）**pre-trained depth network**:  （固定，无优化的参数）

​	**trainer(Trainer).model(GaussianPredictor).models["unidepth_extended"] (UniDepthExtended).unidepth**

​	输入：inputs["color_aug", 0, 0], intrinsics=inputs[("K_src", 0)]

​	输出：深度图(1 * H * W)

（2）**ResNet Encoder**:

​	**trainer(Trainer).model(GaussianPredictor).models["unidepth_extended"] (UniDepthExtended).endcoder(ResnetEncoder)**

​	输入：torch.cat([inputs["color_aug", 0, 0], depth_outs["depth"] / 20.0], dim=1) # 拼接深度图和RGB图（共4 * H * W）

​	输出：features列表

​	如果输入图像尺寸是 **`(B, 3, H, W)`**（`B` 是 batch size，`H, W` 是高和宽），则输出 `features` 的形状如下（以 `ResNet-18/50` 为例）：

| **特征层**    | **来源**                  | **通道数**                         | **空间尺寸 (H, W)** |
| ------------- | ------------------------- | ---------------------------------- | ------------------- |
| `features[0]` | `conv1 + bn1 + relu`      | 64                                 | `(H/2, W/2)`        |
| `features[1]` | `layer1` (ResNet block 1) | 64                                 | `(H/4, W/4)`        |
| `features[2]` | `layer2` (ResNet block 2) | 128                                | `(H/8, W/8)`        |
| `features[3]` | `layer3` (ResNet block 3) | 256 (ResNet-18) / 512 (ResNet-50)  | `(H/16, W/16)`      |
| `features[4]` | `layer4` (ResNet block 4) | 512 (ResNet-18) / 2048 (ResNet-50) | `(H/32, W/32)`      |

features[i]分别保存每层的输出是为了后面 encoder和decoder 残差连接

（3）**ResnetDepthDecoder**（预测深度）

**trainer(Trainer).model(GaussianPredictor).models["unidepth_extended"] (UniDepthExtended).models["depth"] (ResnetDepthDecoder)**

输入：features列表

输出：outputs （outputs[("depth", 0)]： (B * (gaussians_per_pixel - 1), 1, H, W)）

（4）**ResnetDecoder**（预测其他高斯参数）

**trainer(Trainer).model(GaussianPredictor).models["unidepth_extended"] (UniDepthExtended).models["gauss_decoder_"+str(i)] (ResnetDecoder)**

输入：features列表

输出：

```python
out = {
    "gauss_opacity":        (B, gaussians_per_pixel, 1, H, W),
    "gauss_scaling":        (B, gaussians_per_pixel, 3, H, W),
    "gauss_rotation":       (B, gaussians_per_pixel, 4, H, W),
    "gauss_features_dc":    (B, gaussians_per_pixel, 3, H, W),
    "gauss_offset":         (B, gaussians_per_pixel, 3, H, W)  # (可选)
    "gauss_features_rest":  (B, gaussians_per_pixel, 9, H, W)  # (可选)
} # 这里one_gauss_decoder: false，所以gaussians_per_pixel为1
```

（5）**UniDepthExtended**（返回所有预测的高斯参数）

**trainer(Trainer).model(GaussianPredictor).models["unidepth_extended"] (UniDepthExtended)**

输入：imputs

输出：outputs_gauss：

```python
{
    ("K_src", 0): (B, 3, 3),
    ("inv_K_src", 0): (B, 3, 3),
    ("depth", 0): (B * gaussians_per_pixel, 1, H, W),
    "gauss_opacity": (B * gaussians_per_pixel, 1, H, W),
    "gauss_scaling": (B * gaussians_per_pixel, 3, H, W),
    "gauss_rotation": (B * gaussians_per_pixel, 4, H, W),
    "gauss_features_dc": (B * gaussians_per_pixel, 3, H, W),
    "gauss_offset": (B, gaussians_per_pixel, 3, H, W),  # (可选)
    "gauss_features_rest": (B * gaussians_per_pixel, sh_channels, H, W)  # 如果 max_sh_degree > 0
}

```

(6) **GaussianPredicter：**

**trainer(Trainer).model(GaussianPredictor)**

输入： inputs

输出：outputs

```python
{
    ("K_src", 0): (B, 3, 3),
    ("inv_K_src", 0): (B, 3, 3),
    ("depth", 0): (B * gaussians_per_pixel, 1, H, W),
    "gauss_opacity": (B * gaussians_per_pixel, 1, H, W),
    "gauss_scaling": (B * gaussians_per_pixel, 3, H, W),
    "gauss_rotation": (B * gaussians_per_pixel, 4, H, W),
    "gauss_features_dc": (B * gaussians_per_pixel, 3, H, W),
    "gauss_offset": (B, gaussians_per_pixel, 3, H, W),  # (可选)
    "gauss_features_rest": (B * gaussians_per_pixel, sh_channels, H, W), # 如果 max_sh_degree > 0
    # self.process_gt_poses(inputs, outputs)生成：
    "gauss_means"：(B * gaussians_per_pixel, 4, H, W), # 应该是frame_0的相机坐标系下
    ("cam_T_cam", 0, f_i): (B, 4, 4), 
    ("cam_T_cam", f_i, 0): (B, 4, 4),
    # self.render_images(inputs, outputs)生成：
    ("color_gauss", frame_id, scale): (B, 3, H, W),
    ("depth_gauss", frame_id, scale): (B, 1, H, W)  # 如果渲染的时候有的话，就会加上这个属性
}
```

