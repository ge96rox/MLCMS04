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
        z = self.sample_reparameterize(mu, logsigma)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logsigma

    def loss_function(self, x, reconstruction, mu, logsigma):
        mse_loss = nn.MSELoss(reduction='sum')
        kl = self.kl_divergence(mu, logsigma)
        outp = -0.5 * mse_loss(x, reconstruction) - kl
        return outp

    def encoder(self, x):
        x = self.en_linear1(x)
        x = self.en_relu1(x)
        mu = self.en_linear2_mu(x)
        mu = self.en_relu2_mu(mu)
        logsigma = self.en_linear2_logsigma(x)
        logsigma = self.en_relu2_logsigma(logsigma)
        return mu, logsigma

    def sample_reparameterize(self, mu, logsigma):
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

    def gen_sample_data(self, sample_data_size, mu, logsigma):
        distr = torch.distributions.normal.Normal(mu.mean(0), logsigma.mean(0).exp())
        z = distr.sample(torch.Size([sample_data_size]))
        gendata = self.decoder(z)
        return gendata


def show_mnist(nr_row, nr_col, data_loader):
    fig = plt.figure(figsize=(5, 5))
    nr_total = nr_row * nr_col
    for i in range(nr_total):
        idx = np.random.randint(0, len(data_loader))
        ax = fig.add_subplot(nr_row, nr_col, i + 1)
        image = data_loader.dataset[idx][0].view(28, 28).numpy()
        ax.imshow(image)


def show_reconstruct_mnist(nr_row, nr_col, reconstr, y):
    fig = plt.figure(figsize=(5, 5))
    nr_total = nr_row * nr_col
    for i in range(nr_total):
        idx = np.random.randint(0, len(reconstr))
        ax = fig.add_subplot(nr_row, nr_col, i + 1)
        image = reconstr[idx].view(28, 28).detach().numpy()
        ax.imshow(image)
        ax.axis("off")
        ax.set_title("ground truth {0}".format(y[idx].item()))
    plt.show()


def show_generate_mnist(nr_row, nr_col, gendata):
    fig = plt.figure(figsize=(5, 5))
    nr_total = nr_row * nr_col
    for i in range(nr_total):
        idx = np.random.randint(0, len(gendata))
        ax = fig.add_subplot(nr_row, nr_col, i + 1)
        image = gendata[idx].view(28, 28).detach().numpy()
        ax.imshow(image)
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('dataset', train=True, download=True,
                                   transform=torchvision.transforms.ToTensor(),
                                   ), batch_size=batch_size, shuffle=True)
    max_epochs = 1
    show_step = 100
    input_size = 28 * 28
    output_size = 28 * 28
    vae = VAE(256, 256, 256, input_size, output_size)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for epoch in range(max_epochs):
        print("Epoch {0}".format(epoch))
        for i, batch in enumerate(train_loader):
            x, y = batch
            x = x.view(x.shape[0], -1)
            opt.zero_grad()
            reconstruction, mu, logsigma = vae.forward_elbo(x)
            loss = -vae.loss_function(x, reconstruction, mu, logsigma).mean(-1)
            loss.backward()
            opt.step()
            if i % show_step == 0:
                print("loss = {0}".format(loss.item()))
                # show_reconstruct_mnist(2, 2, reconstruction, y)
                gen_data = vae.gen_sample_data(4, mu, logsigma)
                show_generate_mnist(2, 2, gen_data)
