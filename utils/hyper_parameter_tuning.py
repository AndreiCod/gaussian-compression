from sklearn.preprocessing import MinMaxScaler

from scipy.stats import differential_entropy

from math import log2

from omegaconf import OmegaConf

import tinycudann as tcnn
import torch


def get_encoding_size_mb(encoding_config: dict, encoding_dtype: torch.dtype) -> float:
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

    encoding_size_mb = get_encoding_size_mb(encoding_config, dtype)

    while encoding_size_mb > max_size_mb:
        encoding_config["log2_hashmap_size"] -= 1
        encoding_size_mb = get_encoding_size_mb(encoding_config, dtype)

    while (
        encoding_size_mb < max_size_mb
        and encoding_config["log2_hashmap_size"] < max_log2_hashmap_size
    ):
        encoding_config["log2_hashmap_size"] += 1
        encoding_size_mb = get_encoding_size_mb(encoding_config, dtype)
    if encoding_size_mb > max_size_mb:
        encoding_config["log2_hashmap_size"] -= 1
        encoding_size_mb = get_encoding_size_mb(encoding_config, dtype)

    return encoding_config


def get_encoding_utilization(
    X: torch.tensor, encoding_config: dict, encoding_dtype: torch.dtype
) -> float:
    encoding = tcnn.Encoding(
        n_input_dims=3,
        encoding_config=encoding_config,
        dtype=encoding_dtype,
    )
    encoding.params.data = torch.zeros_like(encoding.params.data)
    encoding.train()

    y_encoded = encoding(X)
    grad = torch.autograd.grad(
        y_encoded,
        encoding.params,
        torch.ones_like(y_encoded),
    )

    grad_values = grad[0]
    modified_entries = torch.nonzero(grad_values)

    hash_encoding_utility = modified_entries.nelement() / encoding.params.nelement()

    return hash_encoding_utility


def grid_scale(level: int, log2_per_level_scale: float, base_resolution: int) -> float:
    return base_resolution * 2 ** (level * log2_per_level_scale) - 1.0


def pos_fract(X: torch.Tensor, scale: float) -> torch.Tensor:
    return (X * scale + 0.5).floor().to(torch.int32)


def sort_by_grid_index(X_grid_index: torch.Tensor) -> tuple:
    X_grid_index = torch.cat(
        (torch.arange(X_grid_index.shape[0]).unsqueeze(1), X_grid_index), dim=1
    )
    for dim in range(X_grid_index.shape[1] - 1, 0, -1):
        sorted_indices = torch.argsort(X_grid_index[:, dim], stable=True)
        X_grid_index = X_grid_index[sorted_indices]
    return X_grid_index[:, 0], X_grid_index[:, 1:]


def estimate_level_entropy(
    X: torch.tensor, y: torch.tensor, scale: float
) -> torch.tensor:
    X_grid_index = pos_fract(X, scale)
    sorted_indicies, sorted_X_grid_index = sort_by_grid_index(X_grid_index)

    _, unique_counts = torch.unique(sorted_X_grid_index, return_counts=True, dim=0)
    indicies_groups = torch.split(sorted_indicies, unique_counts.tolist())

    y_entropy = y.clone()

    for group in indicies_groups:
        if group.shape[0] < 5:
            y_entropy[group] = torch.full_like(y_entropy[group], -torch.inf)
            continue

        y_entropy[group] = torch.tensor(
            differential_entropy(
                y_entropy[group].numpy(), method="ebrahimi", axis=0, keepdims=True
            ),
            dtype=torch.float32,
        )

    y_entropy[y_entropy == -torch.inf] = 0
    y_entropy = y_entropy.sum(dim=1)

    return y_entropy


def get_encoding_gradients(
    X: torch.tensor, y: torch.tensor, encoding: tcnn.Encoding
) -> torch.Tensor:
    encoding.params.data = torch.zeros_like(encoding.params.data)
    encoding.train()

    y_encoded = encoding(X)
    grad = torch.autograd.grad(
        y_encoded,
        encoding.params,
        y.unsqueeze(1).repeat(1, y_encoded.shape[1]).to("cuda"),
    )
    grad_values = grad[0]

    return grad_values


def estimate_encoding_entropy(
    X: torch.tensor, y: torch.tensor, encoding_config: dict, encoding_dtype: torch.dtype
) -> torch.tensor:

    y_entropy = torch.zeros(y.shape[0], dtype=torch.float32)

    for level in range(encoding_config["n_levels"]):

        scale = grid_scale(
            level,
            log2(encoding_config["per_level_scale"]),
            encoding_config["base_resolution"],
        )

        level_entropy = estimate_level_entropy(X, y, scale)
        if (level_entropy == 0).all():
            break

        y_entropy += level_entropy

    encoding = tcnn.Encoding(
        n_input_dims=3,
        encoding_config=encoding_config,
        dtype=encoding_dtype,
    )
    encoding.params.data = torch.zeros_like(encoding.params.data)
    encoding.train()

    param_weights = get_encoding_gradients(X, torch.ones_like(y_entropy), encoding).to(
        "cpu"
    )
    param_weights = param_weights.to(torch.float64)
    weighted_entropy = get_encoding_gradients(X, y_entropy, encoding).to("cpu")
    weighted_entropy = weighted_entropy.to(torch.float64)
    params_entropy = weighted_entropy / param_weights
    params_entropy[torch.isnan(params_entropy)] = 0

    encoding_entropy = params_entropy.sum()

    return encoding_entropy


def tune_scaling_for_encoding(
    X: torch.tensor, y: torch.tensor, conf: OmegaConf, scales: list = [1.0, 1.5, 2.0]
) -> tuple:
    encoding_config = dict(conf.encoding_params.encoding_config)
    encoding_dtype: torch.dtype = (
        torch.float32 if conf.encoding_params.use_float32 else torch.float16
    )

    encoding_utilization = get_encoding_utilization(
        X, dict(conf.encoding_params.encoding_config), torch.float32
    )

    best_score = torch.inf

    for scale in scales:
        X_scaler = MinMaxScaler(feature_range=(-scale, scale))
        X_scaled = X_scaler.fit_transform(X)
        X_scaled = torch.tensor(X_scaled, dtype=torch.float32).contiguous()

        encoding_entropy = estimate_encoding_entropy(
            X_scaled, y, encoding_config, encoding_dtype
        )
        encoding_score = encoding_entropy * encoding_utilization
        print(f"Scale: {scale}, Score: {encoding_score}")

        if encoding_score < best_score:
            best_score = encoding_score
            best_scale = scale

    return best_scale, best_score
