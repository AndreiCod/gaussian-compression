import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


def plot_loss(loss_history: np.array, save_path: str):
    fig, ax = plt.subplots()
    best_loss = np.min(loss_history)
    sns.lineplot(x=range(len(loss_history)), y=loss_history, ax=ax)
    ax.axhline(
        best_loss, color="r", linestyle="--", label=f"Best Loss: {best_loss:.4f}"
    )
    ax.set_title("Loss history")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    plt.savefig(save_path)
    plt.close()
