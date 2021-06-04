import numpy as np
import torch
import matplotlib.pyplot as plt
from src.vae import *


def plot_fire_evac_dataset(data):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(data[:, 0], data[:, 1])
    plt.show()


def plot_fire_evac_tensor_dataset(data):
    data = data.detach().numpy()
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(data[:, 0], data[:, 1])
    plt.show()


if __name__ == "__main__":
    pass
