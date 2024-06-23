from math import e
from pathlib import Path
import os
from sys import exception
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from gs.utils.loss_utils import ssim
from gs.lpipsPyTorch import lpips
import json
from tqdm import tqdm
from gs.utils.image_utils import psnr
from argparse import ArgumentParser

from utils import try_gpu


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"
            method_dir = test_dir
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips(renders[idx], gts[idx], net_type="vgg"))

            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            full_dict[scene_dir].update(
                {
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item(),
                }
            )
            per_view_dict[scene_dir].update(
                {
                    "SSIM": {
                        name: ssim
                        for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)
                    },
                    "PSNR": {
                        name: psnr
                        for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)
                    },
                    "LPIPS": {
                        name: lp
                        for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)
                    },
                }
            )

            with open(scene_dir + "/results.json", "w") as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", "w") as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            print(e)


if __name__ == "__main__":
    device = try_gpu()
    torch.cuda.set_device(device)

    parser = ArgumentParser()
    parser.add_argument(
        "--model_paths", "-m", required=True, nargs="+", type=str, default=[]
    )
    args = parser.parse_args()
    evaluate(args.model_paths)
