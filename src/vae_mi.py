import matplotlib.patches as patches
from src.vae import *


def plot_fire_evac_dataset(data, scale):
    data = data * scale
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    rect = patches.Rectangle((130, 50), 20, 20, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.scatter(data[:, 0], data[:, 1])
    ax.axis('equal')
    plt.show()
    return ax


def plot_fire_evac_tensor_dataset(data, scale):
    data = data.detach().numpy()
    data = data * scale
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    rect = patches.Rectangle((130, 50), 20, 20, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.scatter(data[:, 0], data[:, 1])
    ax.axis('equal')
    plt.show()
    return ax


def cal_nr_persons(gen_data, scale):
    gen_data_scale = gen_data * scale
    x_satisfy = np.logical_and(gen_data_scale[:, 0] >= 130, gen_data_scale[:, 0] <= 150)
    y_satisfy = np.logical_and(gen_data_scale[:, 1] >= 50, gen_data_scale[:, 1] <= 70)
    xy_satisfy = np.logical_and(x_satisfy, y_satisfy)
    return np.sum(xy_satisfy)


if __name__ == "__main__":
    pass
