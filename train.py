import sys, os
import time
import json
from omegaconf import OmegaConf
from argparse import ArgumentParser

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import gc
from datetime import datetime
from tqdm import tqdm
import tinycudann as tcnn

from utils import try_gpu, set_seed
from utils.point_cloud import load_ply, save_ply
from utils.config import load_config
from utils.plots import plot_loss
from utils.hyper_parameter_tuning import tune_encoding_config
from models import CompressionNet


def loss(output, target):
    l2_loss = nn.MSELoss()
    return l2_loss(output, target)


def train_epoch(net, encoding, train_iter, loss, optimizer, device):
    # Set the model to training mode
    net.train()
    # Sum of training loss
    total_loss = 0
    for X, y in train_iter:
        # Compute gradients and update parameters
        X, y = X.to(device), y.to(device)
        y_hat = net(X, encoding(X))
        l = loss(y_hat, y)
        # Using PyTorch built-in optimizer & loss criterion
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        total_loss += float(l)
    # Return training loss
    return float(total_loss) / len(train_iter)


def train(
    net,
    encoding,
    train_iter,
    num_epochs,
    lr,
    device,
    display_interval=10,
    early_stopping=None,
):
    loss_history = []

    training_time = time.time()

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    net = net.to(device)
    net.apply(init_weights)

    optimizer = Adam([*net.parameters(), *encoding.parameters()], lr=lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[250, 500, 750, 1000], gamma=0.56234
    )

    progress_bar = tqdm(range(num_epochs), unit="epoch", desc="Training")

    for epoch in range(num_epochs):
        # Empty cache
        gc.collect()
        torch.cuda.empty_cache()

        epoch_loss = train_epoch(net, encoding, train_iter, loss, optimizer, device)
        loss_history.append(epoch_loss)

        if epoch % display_interval == 0:
            progress_bar.set_postfix({"Loss": f"{epoch_loss:.{7}f}"})
            progress_bar.update(display_interval)

        scheduler.step()

        if early_stopping is not None and len(loss_history) > early_stopping:
            if loss_history[-early_stopping] < min(loss_history):
                print(f"Early stopping at epoch {epoch}")
                break

    progress_bar.close()

    training_time = time.time() - training_time

    return loss_history, training_time


def train_iteration(
    point_cloud_path: str, conf: OmegaConf, conf_3dgs: OmegaConf, output_path: str
):
    # Load 3dgs point cloud
    X, y = load_ply(point_cloud_path, conf_3dgs.model_params.sh_degree)
    X_size_mb = X.element_size() * X.nelement() / 1024**2
    y_size_mb = y.element_size() * y.nelement() / 1024**2
    size_3dgs_pc_mb = X_size_mb + y_size_mb

    # Normalize data
    X_scaler = MinMaxScaler(feature_range=tuple(conf.data_params.X_scaling))
    X = X_scaler.fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32).contiguous()

    y_scaler = MinMaxScaler(feature_range=tuple(conf.data_params.y_scaling))
    y = y_scaler.fit_transform(y)
    y = torch.tensor(y, dtype=torch.float32).contiguous()

    # Create dataloader
    dataset = torch.utils.data.TensorDataset(X, y)
    data_loader = DataLoader(
        dataset,
        batch_size=conf.model_params.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )

    # Tune encoding parameters
    encoding_dtype = (
        torch.float32 if conf.encoding_params.use_float32 else torch.float16
    )
    max_encoding_size_mb = (
        size_3dgs_pc_mb + X_size_mb * conf.model_params.compression_ratio
    ) / conf.model_params.compression_ratio

    conf.encoding_params.encoding_config = tune_encoding_config(
        dict(conf.encoding_params.encoding_config),
        encoding_dtype,
        max_encoding_size_mb,
    )

    # Create encoding model
    encoding = tcnn.Encoding(
        n_input_dims=3,
        encoding_config=dict(conf.encoding_params.encoding_config),
        dtype=encoding_dtype,
    )
    encoding.train()

    # Create CompressionNet model
    net = CompressionNet(
        input_size=3,
        hash_encoding_size=encoding.n_output_dims,
        output_size=y.shape[1],
        hidden_size=conf.model_params.hidden_size,
        num_hidden_layers=conf.model_params.num_hidden_layers,
        num_freq=conf.model_params.num_freq,
        output_activation=[conf.model_params.output_activation] * y.shape[1],
        output_split_size=[
            1,
            3,
            3 * (conf_3dgs.model_params.sh_degree + 1) ** 2 - 3,
            3,
            4,
        ],
        dtype=torch.float32,
    ).to(device)
    net.train()

    # Calculate compression ratio
    encoding_size_mb = (
        encoding.params.nelement() * encoding.params.element_size() / 1024**2
    )
    net_size_mb = sum(
        p.element_size() * p.nelement() / 1024**2 for p in net.parameters()
    )
    model_size_mb = encoding_size_mb + net_size_mb
    compression_ratio = size_3dgs_pc_mb / (model_size_mb + X_size_mb)

    # Train model
    loss_history, training_time = train(
        net,
        encoding,
        data_loader,
        conf.model_params.num_epochs,
        conf.model_params.learning_rate,
        device,
        conf.model_params.display_interval,
        conf.model_params.early_stopping,
    )

    # Save results
    directory = output_path
    os.makedirs(directory, exist_ok=True)
    results: dict = {
        "compression_rate": compression_ratio,
        "model_size_mb": model_size_mb,
        "training_time": training_time,
        "best_loss": min(loss_history),
    }
    with open(f"{directory}/results_compression.json", "w") as f:
        json.dump(results, f, indent=4)
    plot_loss(loss_history, os.path.join(directory, "loss.png"))
    np.save(os.path.join(directory, "loss_history.npy"), np.array(loss_history))
    OmegaConf.save(conf, os.path.join(directory, "compression_config.yaml"))

    # Save models
    net_path = os.path.join(directory, "net.pickle")
    encoding_path = os.path.join(directory, "encoding.pickle")
    net.eval()
    encoding.eval()
    torch.save(net.state_dict(), net_path)
    torch.save(encoding.state_dict(), encoding_path)

    # Save point cloud
    point_cloud_path = os.path.join(directory, "point_cloud/point_cloud.ply")
    net.eval()
    net.load_state_dict(torch.load(net_path))
    net.to("cpu")
    encoding.eval()
    encoding.load_state_dict(torch.load(encoding_path))
    encoding.to(device)
    with torch.no_grad():
        y_pred = torch.tensor(
            y_scaler.inverse_transform(
                net(X.to("cpu"), encoding(X.to(device)).detach().cpu()).detach()
            ),
            dtype=torch.float32,
        )
    X = torch.tensor(X_scaler.inverse_transform(X), dtype=torch.float32)

    save_ply(X, y_pred, point_cloud_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file",
    )
    args = parser.parse_args(sys.argv[1:])

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    conf, conf_3dgs = load_config(args.config)

    set_seed(conf.general_params.seed)
    device = try_gpu()

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

    # Set output path
    output_path: str = os.path.join(
        conf.general_params.gaussian_model_path, "compression"
    )
    os.makedirs(
        output_path,
        exist_ok=True,
    )
    OmegaConf.save(
        conf_3dgs,
        os.path.join(
            output_path,
            "3dgs_config.yaml",
        ),
    )

    for iteration in iterations:
        print(f"Processing iteration {iteration}...")
        print("Loading point cloud...")
        input_point_cloud_path = os.path.join(
            point_cloud_dir_path, iteration, "point_cloud.ply"
        )
        iteration_output_path = os.path.join(
            output_path,
            iteration,
        )
        os.makedirs(
            iteration_output_path,
            exist_ok=True,
        )

        train_iteration(
            input_point_cloud_path, conf.copy(), conf_3dgs.copy(), iteration_output_path
        )
