import os, sys
from argparse import ArgumentParser

from omegaconf import OmegaConf

from utils.config import load_compression_config


def batch_eval(config_path: str, models_dir: str):
    conf = load_compression_config(config_path)
    for model in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model)
        if not os.path.isdir(model_path):
            continue
        print(f"Evaluating model {model_path}")

        conf.general_params.gaussian_model_path = model_path
        OmegaConf.save(conf, config_path)

        os.system(f"python train.py --config {config_path}")

        # Set which iterations to process
        point_cloud_dir_path = os.path.join(
            conf.general_params.gaussian_model_path, "point_cloud"
        )
        iterations = [
            folder
            for folder in os.listdir(point_cloud_dir_path)
            if os.path.isdir(os.path.join(point_cloud_dir_path, folder))
        ]
        iterations = sorted(iterations, key=lambda x: int(x.split("_")[-1]))
        if conf.general_params.use_best_iteration == True:
            iterations = iterations[-1:]

        for iteration in iterations:
            iteration_int = int(iteration.split("_")[-1])
            iteration_path = os.path.join(model_path, "compression", iteration)
            os.system(
                f"python render.py --model_path {model_path} --iteration {iteration_int}"
            )
            os.system(f"python metrics.py --model_path {iteration_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Path to the models directory",
    )
    args = parser.parse_args(sys.argv[1:])

    batch_eval(args.config, args.models_dir)
