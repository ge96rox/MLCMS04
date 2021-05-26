import numpy as np
import matplotlib.pyplot as plt


def gen_trigonom_data(n):
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
    dist = data[np.newaxis, :, :] - data[:, np.newaxis, :]
    dist_matrix = np.linalg.norm(dist, axis=-1)
    eps = 0.05 * np.max(dist_matrix)
    w = np.exp(-dist_matrix / eps)
    p = np.diag(np.sum(w, axis=-1))
    k = np.linalg.inv(p) @ w @ np.linalg.inv(p)
    q = np.diag(np.sum(k, axis=-1))
    t_hat = np.sqrt(np.linalg.inv(q)) @ k @ np.sqrt(np.linalg.inv(q))
    eig_values, eig_vectors = np.linalg.eigh(t_hat)
    a_pc = eig_values[-nr_component:]
    v_pc = eig_vectors[:, -nr_component:]
    lambda_pc = np.sqrt(np.power(a_pc, 1 / eps))
    phi_pc = np.linalg.inv(np.sqrt(q)) @ v_pc
    return lambda_pc, phi_pc, dist_matrix


def plot_part_one_eigenfunction(nr_component, time, phi_pc, lambda_pc):
    for comp in range(nr_component):
        plt.plot(time, phi_pc[:, comp], label='Eigenvalue {0} '.format(lambda_pc[comp]))
        plt.xlabel("time")
        plt.ylabel("eigen vector")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    pass
