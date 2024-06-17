import os
from dataclasses import dataclass
from argparse import Namespace


class AbstractGaussianParams:
    def extract(self, args: Namespace) -> dict:
        for arg_key, arg_value in vars(args).items():
            if arg_key in vars(self).keys():
                setattr(self, arg_key, arg_value)
        return vars(self).copy()


@dataclass
class GaussianModelParams(AbstractGaussianParams):
    def __init__(self):
        self.sh_degree = 3
        self.source_path = ""
        self.model_path = ""
        self.images = "images"
        self.resolution = -1
        self.white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__()


@dataclass
class GaussianPipelineParams(AbstractGaussianParams):
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__()


@dataclass
class GaussianOptimizationParams(AbstractGaussianParams):
    def __init__(self):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        super().__init__()


def extract_gaussian_params(cfg_path: str) -> dict:
    cfgfile_string: str = "Namespace()"

    try:
        cfgfilepath: str = os.path.join(cfg_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string: str = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile: Namespace = eval(cfgfile_string)

    gmp = GaussianModelParams()
    gop = GaussianOptimizationParams()
    gpp = GaussianPipelineParams()

    gmp.extract(args_cfgfile)
    gop.extract(args_cfgfile)
    gpp.extract(args_cfgfile)

    gaussian_args: dict = {
        "model_params": vars(gmp),
        "optimization_params": vars(gop),
        "pipeline_params": vars(gpp),
    }

    return gaussian_args
