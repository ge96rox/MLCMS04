import numpy as np
import torch
import matplotlib.pyplot as plt
from src.vae import *


def plot_fire_evac_dataset(data):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(data[:, 0], data[:, 1])
    plt.show()

def plot_fire_evac_tensor_dataset(data):
    data = data.detach().numpy()
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(data[:, 0], data[:, 1])
    plt.show()
  


if __name__ == "__main__":
    train_dataset = np.load("./datasets/FireEvac_train_set.npy")
    test_dataset = np.load("./datasets/FireEvac_test_set.npy")

    max_train_dataset = np.max(train_dataset, axis=0)
    min_train_dataset = np.min(train_dataset, axis=0)
    max_test_dataset = np.max(test_dataset, axis=0)
    min_test_dataset = np.min(test_dataset, axis=0)

    range_train_dataset = max_train_dataset - min_train_dataset
    range_test_dataset = max_test_dataset - min_test_dataset
    train_dataset = (train_dataset / range_train_dataset).astype('float32')
    test_dataset = (test_dataset / range_test_dataset).astype('float32')

    test_size = len(test_dataset)
    train_size = len(train_dataset)

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=batch_size, shuffle=True)

    max_epochs = 2
    show_step = 100
    input_size = 2
    output_size = 2
    latent_dim = 2
    en_hidden = 256
    de_hidden = 256
    gendata_size = 100
    vae = VAE(en_hidden, latent_dim, de_hidden, input_size, output_size)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    loss_train = []
    loss_test = []
    loss_epoch = 0
    i = 0
    for epoch in range(max_epochs):
        print("Epoch {0}".format(epoch))
        for batch in train_loader:
            x = batch
            opt.zero_grad()
            reconstruction, mu, sigma = vae.forward_elbo(x)
            loss = 1 / batch_size * -vae.loss_function(x, reconstruction, mu, sigma).mean(-1)
            loss.backward()
            opt.step()
            loss_epoch += loss
            i += 1

        loss_epoch = loss_epoch / i
        print("loss = {0}".format(loss_epoch.item()))
        loss_train.append(loss_epoch)
        loss_epoch = 0
        i = 0
        for batch_test in test_loader:
            x_test = batch_test
            with torch.no_grad():
                reconstruction_test, mu_test, sigma_test = vae.forward_elbo(x_test)
                loss = 1 / test_size * -vae.loss_function(x_test, reconstruction_test, mu_test, sigma_test).mean(-1)
                loss_test.append(loss)
            if epoch:
                print("plot reconstruction distribution")
                # plot_fire_evac_dataset(mu_test)
                print("generate synthetic data")
                gen_data = vae.gen_sample_data(gendata_size, latent_dim)
                # plot_fire_evac_dataset(gen_data)