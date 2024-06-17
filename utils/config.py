from omegaconf import OmegaConf
from arguments import CompressionParams, extract_gaussian_params


def load_config(config_path: str) -> OmegaConf:
    conf = OmegaConf.structured(CompressionParams())
    conf = OmegaConf.merge(conf, OmegaConf.load(config_path))

    conf_3dgs = OmegaConf.create(
        extract_gaussian_params(conf.general_params.gaussian_model_path)
    )

    return conf, conf_3dgs
