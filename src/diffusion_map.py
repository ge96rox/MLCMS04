import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from src.pca import *


def gen_trigonom_data(n):
    """function that generate trigonometric data

    Parameters
    ----------
    n: int
        number of generated sample
    """
    time = np.linspace(0, 2 * np.pi, n + 2)[1:n + 1]
    x = np.cos(time)
    y = np.sin(time)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.subplots(1, 1)
    ax.scatter(x, y, s=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Task 1 Part 1 data set")
    plt.show()
    trig_data = np.vstack((x, y)).T
    return trig_data, time


def diffusion_map(data, nr_component):
    """algorithm that calculate eigenvectors and eigenvalues for diffusion maps

    Parameters
    ----------
    data: np.ndarray
        dataset
    nr_component: int
        number of required principle components

    Returns
    ----------
    lambda_pc:
       lambda of principle components
    phi_pc:
        eigenvectors of principle components
    dist_matrix:
        distrance matrix
    """
    dist = data[np.newaxis, :, :] - data[:, np.newaxis, :]
    dist_matrix = np.linalg.norm(dist, axis=-1)
    eps = 0.05 * np.max(dist_matrix)
    w = np.exp(-dist_matrix**2 / eps)
    p = np.diag(np.sum(w, axis=-1))
    k = np.linalg.inv(p) @ w @ np.linalg.inv(p)
    q = np.diag(np.sum(k, axis=-1))
    t_hat = np.sqrt(np.linalg.inv(q)) @ k @ np.sqrt(np.linalg.inv(q))
    eig_values, eig_vectors = np.linalg.eigh(t_hat)
    a_pc = eig_values[-nr_component-1:]
    v_pc = eig_vectors[:, -nr_component-1:]
    lambda_pc = np.sqrt(np.power(a_pc, 1 / eps))
    phi_pc = np.linalg.inv(np.sqrt(q)) @ v_pc
    return lambda_pc, phi_pc, dist_matrix


def gen_swiss_roll_data(nr_sample, noise):
    """function that generate swiss roll data

    Parameters
    ----------
    nr_sample: int
        number of required data
    noise:
        The standard deviation of the gaussian noise
    Returns
    ----------
    tuple:
        swiss roll data
    """
    return make_swiss_roll(nr_sample, noise)


def plot_eigenfunction(nr_component, time, phi_pc, lambda_pc):
    """function that plot time against eigenfucntion

    Parameters
    ----------
    nr_component: int
        number of required principle components
    time: np.ndarray
        time
    lambda_pc: np.ndarray
       lambda of principle components
    phi_pc: np.ndarray
        eigenvectors of principle components
    """
    for comp in range(nr_component+1):
        plt.plot(time, phi_pc[:, comp], label='Eigenvalue {0} '.format(lambda_pc[comp]))
        plt.xlabel("time")
        plt.ylabel("eigen vector")
        plt.legend()
        plt.show()


def plot_swiss_roll(data, t, text):
    """function that generate swiss roll data

    Parameters
    ----------
    data: np.ndarray
        data
    t: np.ndarray
        color map
    text: str
        text to set title
    Returns
    ----------
    tuple:
        swiss roll data
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=t, cmap=plt.cm.Spectral)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title("Swiss-Roll data manifold {0}".format(text))
    ax.view_init(10, 70)
    return fig


if __name__ == "__main__":
    pass
