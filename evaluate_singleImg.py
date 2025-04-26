import os
# import json
import hydra
import torch
import numpy as np
import torchvision.transforms as T

# from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
# from matplotlib import pyplot as plt
# import torchvision.transforms.functional as TF

from models.model import GaussianPredictor, to_device
# from evaluation.evaluator import Evaluator
# from datasets.util import create_datasets
# from misc.util import add_source_frame_id
from misc.visualise_3d import save_ply
from PIL import Image

def get_model_instance(model):
    """
    unwraps model from EMA object
    """
    return model.ema_model if type(model).__name__ == "EMA" else model

def get_intrinsic_matrix(width, height, fx=None, fy=None, cx=None, cy=None):
    if fx is None: fx = width * 0.5
    if fy is None: fy = height * 0.5
    if cx is None: cx = width * 0.5
    if cy is None: cy = height * 0.5

    K = torch.tensor([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=torch.float32)
    return K

def make_inputs(cfg, image_path):
    pad_border_fn = T.Pad((cfg.dataset.pad_border_aug, cfg.dataset.pad_border_aug))
    to_tensor = T.ToTensor()

    raw_image = Image.open(image_path).convert("RGB")
    color = pad_border_fn(raw_image)
    color_aug = pad_border_fn(raw_image)

    color = to_tensor(color).unsqueeze(0)
    color_aug = to_tensor(color_aug).unsqueeze(0)

    (width, height) = raw_image.size
    K = get_intrinsic_matrix(width, height)

    input_dict = {
        ("frame_id", 0): 0,
        ("K_tgt", 0): K,
        ("color", 0, 0): color,
        ("color_aug", 0, 0): color_aug,
    }

    return input_dict


def evaluate(model, cfg, evaluator=None, dataloader=None, device=None, save_vis=False):
    model_model = get_model_instance(model)
    model_model.set_eval()
    out_out_dir = Path("./output")
    image_path = r"testData/110133333.jpg"
    score_dict = {}
    # match cfg.dataset.name:
    #     case "re10k" | "nyuv2":
    #         # override the frame idxs used for eval
    #         target_frame_ids = [1, 2, 3]
    #         eval_frames = ["src", "tgt5", "tgt10", "tgt_rand"]
    #         for fid, target_name in zip(add_source_frame_id(target_frame_ids),
    #                                     eval_frames):
    #             score_dict[fid] = { "ssim": [],
    #                                 "psnr": [],
    #                                 "lpips": [],
    #                                 "name": target_name }
                                    
    # dataloader_iter = iter(dataloader)
    # for k in tqdm([i for i in range(len(dataloader.dataset) // cfg.data_loader.batch_size)]):
    #     try:
    #         inputs = next(dataloader_iter)
    #     except Exception as e:
    #         if cfg.dataset.name=="re10k":
    #             if cfg.dataset.test_split in ["pixelsplat_ctx1",
    #                                           "pixelsplat_ctx2",
    #                                           "latentsplat_ctx1",
    #                                           "latentsplat_ctx2"]:
    #                 print("Failed to read example {}".format(k))
    #                 continue
    #         raise e
    if True:
        if True:
            # out_dir = Path("/work/cxzheng/3D/splatvideo/eldar/visual_results/images")
            # out_dir.mkdir(exist_ok=True)
            # print(f"saving images to: {out_dir.resolve()}")
            # seq_name = inputs[("frame_id", 0)][0].split("+")[1]
            # out_out_dir = out_dir / seq_name
            # out_out_dir.mkdir(exist_ok=True)
            # out_pred_dir = out_out_dir / f"pred"
            # out_pred_dir.mkdir(exist_ok=True)
            # out_gt_dir = out_out_dir / f"gt"
            # out_gt_dir.mkdir(exist_ok=True)
            out_dir_ply = out_out_dir / "ply"
            out_dir_ply.mkdir(exist_ok=True, parents=True)

        inputs = make_inputs(cfg, image_path)
        with torch.no_grad():
            if device is not None:
                to_device(inputs, device)
            # inputs["target_frame_ids"] = target_frame_ids
            outputs = model(inputs)

        if True:
        # for f_id in score_dict.keys():
        #     pred = outputs[('color_gauss', f_id, 0)]
        #     if cfg.dataset.name == "dtu":
        #         gt = inputs[('color_orig_res', f_id, 0)]
        #         pred = TF.resize(pred, gt.shape[-2:])
        #     else:
        #         gt = inputs[('color', f_id, 0)]
        #     # should work in for B>1, however be careful of reduction
        #     out = evaluator(pred, gt)
            # if save_vis:
            if True:
                save_ply(outputs, out_dir_ply / f"p.ply", gaussians_per_pixel=model.cfg.model.gaussians_per_pixel)
            #     pred = pred[0].clip(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
            #     gt = gt[0].clip(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
            #     plt.imsave(str(out_pred_dir / f"{f_id:03}.png"), pred)
            #     plt.imsave(str(out_gt_dir / f"{f_id:03}.png"), gt)
            # for metric_name, v in out.items():
            #     score_dict[f_id][metric_name].append(v)

    # metric_names = ["psnr", "ssim", "lpips"]
    # score_dict_by_name = {}
    # for f_id in score_dict.keys():
    #     score_dict_by_name[score_dict[f_id]["name"]] = {}
    #     for metric_name in metric_names:
    #         # compute mean
    #         score_dict[f_id][metric_name] = sum(score_dict[f_id][metric_name]) / len(score_dict[f_id][metric_name])
    #         # original dict has frame ids as integers, for json out dict we want to change them
    #         # to the meaningful names stored in dict
    #         score_dict_by_name[score_dict[f_id]["name"]][metric_name] = score_dict[f_id][metric_name]

    # for metric in metric_names:
    #     vals = [score_dict_by_name[f_id][metric] for f_id in eval_frames]
    #     print(f"{metric}:", np.mean(np.array(vals)))

    # return score_dict_by_name


@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None
)
def main(cfg: DictConfig):
    """
    cfg = {
        "hydra.run.dir": "$1",
        "hydra.job.chdir": True,
        "+experiment": "layered_re10k",
        "+dataset.crop_border": True,
        "dataset.test_split_path": "splits/re10k_mine_filtered/test_files.txt",
        "model.depth.version": "v1",
        "++eval.save_vis": False
    }

    """
    cfg.dataset.scale_pose_by_depth = False
    print("current directory:", os.getcwd())
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    os.chdir(output_dir)
    print("Working dir:", output_dir)

    cfg.data_loader.batch_size = 1
    cfg.data_loader.num_workers = 1
    cfg.model.gaussian_rendering = False
    model = GaussianPredictor(cfg)
    device = torch.device("cuda:0")
    model.to(device)
    if (model.checkpoint_dir()).exists():
        # resume training
        ckpt_dir = model.checkpoint_dir()
        model.load_model(ckpt_dir, ckpt_ids=0)
    
    # evaluator = Evaluator(crop_border=cfg.dataset.crop_border)
    # evaluator.to(device)

    split = "test"
    save_vis = cfg.eval.save_vis
    # dataset, dataloader = create_datasets(cfg, split=split)
    # score_dict_by_name = evaluate(model, cfg, evaluator, dataloader, 
    #                               device=device, save_vis=save_vis)
    score_dict_by_name = evaluate(model, cfg,
                                  device=device, save_vis=save_vis)
    
    # print(json.dumps(score_dict_by_name, indent=4))
    # if cfg.dataset.name=="re10k":
    #     with open("metrics_{}_{}_{}.json".format(cfg.dataset.name, split, cfg.dataset.test_split), "w") as f:
    #         json.dump(score_dict_by_name, f, indent=4)
    # with open("metrics_{}_{}.json".format(cfg.dataset.name, split), "w") as f:
    #     json.dump(score_dict_by_name, f, indent=4)
    

if __name__ == "__main__":
    main()
