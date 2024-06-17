from dataclasses import MISSING, dataclass, field
from argparse import ArgumentParser, Namespace
from math import e

from sympy import field
from torch import seed


class AbstractArgumentParser:
    def __init__(self, parser: ArgumentParser, name: str):
        group = parser.add_argument_group(name)

        for key, value in vars(self).items():
            if isinstance(value, bool):
                group.add_argument(f"--{key}", action="store_true", default=value)
            group.add_argument(f"--{key}", type=type(value), default=value)

    def extract(self, args: Namespace = None) -> dict:
        if args is None:
            return vars(self).copy()

        for arg_key, arg_value in vars(args).items():
            if arg_key in vars(self).keys():
                setattr(self, arg_key, arg_value)
        return vars(self).copy()


@dataclass
class AbstractDataClass:
    def extract(self) -> dict:
        params_dict: dict = {}
        for arg_key, arg_value in vars(self).items():
            if isinstance(arg_value, AbstractDataClass):
                params_dict[arg_key] = arg_value.extract()
            else:
                params_dict[arg_key] = arg_value
        return params_dict


@dataclass
class GeneralParams(AbstractDataClass):
    gaussian_model_path: str = ""
    use_best_iteration: bool = True
    seed: int = 42


@dataclass
class ModelParams(AbstractDataClass):
    compression_ratio: float = 4.0
    batch_size: int = 2**17
    hidden_size: int = 256
    num_hidden_layers: int = 8
    num_freq: int = 10
    output_activation: str = "sigmoid"
    num_epochs: int = 2000
    learning_rate: float = 5e-4
    display_interval: int = 1
    early_stopping: int = 100


@dataclass
class EncodingParams(AbstractDataClass):
    encoding_config: dict
    use_float32: bool = True

    def __init__(self):
        self.encoding_config = {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": 16,
            "n_features_per_level": 4,
            "log2_hashmap_size": 18,
            "base_resolution": 32,
            "per_level_scale": 2.0,
            "interpolation": "Linear",
        }


@dataclass
class DataParams(AbstractDataClass):
    X_scaling: tuple = (-1, 1)
    y_scaling: tuple = (0.05, 0.95)


@dataclass
class CompressionParams(AbstractDataClass):
    general_params: GeneralParams
    model_params: ModelParams
    encoding_params: EncodingParams
    data_params: DataParams

    def __init__(self):
        self.general_params = GeneralParams()
        self.model_params = ModelParams()
        self.encoding_params = EncodingParams()
        self.data_params = DataParams()
