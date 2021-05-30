import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os



class VAE(nn.Module):
    def __init__(self, en_hidden1, en_hidden2, de_hidden1, input_size, output_size):
        super().__init__()
        self.en_linear1 = nn.Linear(input_size, en_hidden1)
        self.en_relu1 = nn.ReLU()
        self.en_linear2_mu = nn.Linear(en_hidden1, en_hidden2)
        self.en_relu2_mu = nn.ReLU()
        self.en_linear2_logsigma = nn.Linear(en_hidden1, en_hidden2)
        self.en_relu2_logsigma = nn.ReLU()
        self.de_linear1 = nn.Linear(en_hidden2, de_hidden1)
        self.de_relu1 = nn.ReLU()
        self.de_linear2 = nn.Linear(de_hidden1, output_size)
        self.de_relu2 = nn.ReLU()

    def forward_elbo(self, x):
        mu, logsigma = self.encoder(x)
        z = self.sample(mu, logsigma)
        loss = nn.MSELoss(reduction='sum')
        phi = self.decoder(z)
        kl = self.kl_divergence(mu, logsigma)
        outp = -loss(x, phi) - kl
        return outp

    def encoder(self, x):
        x = self.en_linear1(x)
        x = self.en_relu1(x)
        mu = self.en_linear2_mu(x)
        mu = self.en_relu2_mu(mu)
        logsigma = self.en_linear2_logsigma(x)
        logsigma = self.en_relu2_logsigma(logsigma)
        return mu, logsigma

    def sample(self, mu, logsigma):
        epsilon = torch.empty_like(mu).normal_(0., 1.)
        return epsilon * logsigma.exp() + mu

    def decoder(self, z):
        z = self.de_linear1(z)
        z = self.de_relu1(z)
        z = self.de_linear2(z)
        z = self.de_relu2(z)
        return z

    def kl_divergence(self, mu, logsigma):
        return 0.5 * (mu.pow(2) + torch.exp((2 * logsigma)) - 2 * logsigma - 1).sum(-1)


def show_mnist(nr_row, nr_col, data_loader):
    fig = plt.figure(figsize=(1.5, 2))
    nr_total = nr_row * nr_col
    for i in range(nr_total):
        idx = np.random.randint(0, len(data_loader))
        ax = fig.add_subplot(nr_row, nr_col, i + 1)
        image = data_loader.dataset[idx][0].view(28, 28).numpy()
        ax.imshow(image)


if __name__ == "__main__":
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('dataset', train=True, download=True,
                                   transform=torchvision.transforms.ToTensor(),
                                   ), batch_size=batch_size, shuffle=True)
    show_mnist(2, 2, train_loader)
