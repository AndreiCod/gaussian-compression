import tinycudann as tcnn
import torch


def calculate_encoding_size_mb(
    encoding_config: dict, encoding_dtype: torch.dtype
) -> float:
    encoding = tcnn.Encoding(
        n_input_dims=3,
        encoding_config=encoding_config,
        dtype=encoding_dtype,
    )
    return encoding.params.element_size() * encoding.params.nelement() / 1024**2


def tune_encoding_config(
    encoding_config: dict,
    dtype: torch.dtype,
    max_size_mb: float,
    max_log2_hashmap_size: int,
) -> dict:

    encoding_size_mb = calculate_encoding_size_mb(encoding_config, dtype)

    while encoding_size_mb > max_size_mb:
        encoding_config["log2_hashmap_size"] -= 1
        encoding_size_mb = calculate_encoding_size_mb(encoding_config, dtype)

    while (
        encoding_size_mb < max_size_mb
        and encoding_config["log2_hashmap_size"] < max_log2_hashmap_size
    ):
        encoding_config["log2_hashmap_size"] += 1
        encoding_size_mb = calculate_encoding_size_mb(encoding_config, dtype)
    if encoding_size_mb > max_size_mb:
        encoding_config["log2_hashmap_size"] -= 1
        encoding_size_mb = calculate_encoding_size_mb(encoding_config, dtype)

    return encoding_config
