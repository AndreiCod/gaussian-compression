import sys, os
import json
import torch
import numpy as np
from PIL import Image
from gs.scene import Scene
from tqdm import tqdm
from os import makedirs
from gs.gaussian_renderer import render, GaussianModel
import torchvision
from argparse import ArgumentParser
from gs.arguments import PipelineParams, ModelParams, get_combined_args
from gs.scene.dataset_readers import CameraInfo


def render_set(
    model_path: str,
    name: str,
    iteration: int,
    views: list,
    gaussians: GaussianModel,
    pipe,
    background: torch.Tensor,
):
    render_path = os.path.join(
        model_path, "compression", f"iteration_{iteration}", name, "renders"
    )
    gts_path = os.path.join(
        model_path, "compression", f"iteration_{iteration}", name, "gt"
    )

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name} set")):
        rendering = render(view, gaussians, pipe, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )


def render_sets(model, iteration: int, pipe):
    gaussians = GaussianModel(model.sh_degree)
    scene = Scene(model, gaussians, iteration, shuffle=False)
    scene.gaussians.load_ply(
        os.path.join(
            model.model_path,
            "compression",
            f"iteration_{scene.loaded_iter}",
            "point_cloud",
            "point_cloud.ply",
        )
    )

    bg_color = [1, 1, 1] if model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_set(
        model.model_path,
        "train",
        scene.loaded_iter,
        scene.getTrainCameras(),
        scene.gaussians,
        pipe,
        background,
    )

    if model.eval == True:
        render_set(
            model.model_path,
            "test",
            scene.loaded_iter,
            scene.getTestCameras(),
            scene.gaussians,
            pipe,
            background,
        )


if __name__ == "__main__":
    # model_path is a required argument
    parser = ArgumentParser()
    parser.add_argument("--iteration", default=-1, type=int)
    pipeline = PipelineParams(parser)
    model = ModelParams(parser, sentinel=True)
    args = get_combined_args(parser)
    args.source_path = os.path.join(
        os.getcwd(), "datasets", os.path.basename(args.model_path)
    )

    print(f"Rendering model {args.model_path} at iteration {args.iteration}...")

    render_sets(model.extract(args), args.iteration, pipeline.extract(args))
