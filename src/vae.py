import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt



class VAE(nn.Module):
    def __init__(self, en_hidden, latent_space, de_hidden, input_size, output_size):
        """Initialize the VAE model

        Parameters
        ----------
        en_hidden : int
            hidden dimension of encoder
        latent_space: int
            dimension of the observed data
        de_hidden: int
            hidden dimension of decoder
        input_size: int
            input dimension
        output_size: int
            output dimension
        """
        super().__init__()
        self.en_linear1 = nn.Linear(input_size, en_hidden)
        self.en_relu1 = nn.ReLU()
        self.en_linear2 = nn.Linear(en_hidden, en_hidden)
        self.en_relu2 = nn.ReLU()
        self.en_linear3_mu = nn.Linear(en_hidden, latent_space)
        self.en_linear3_sigma = nn.Linear(en_hidden, latent_space)
        self.de_linear1 = nn.Linear(latent_space, de_hidden)
        self.de_relu1 = nn.ReLU()
        self.de_linear2 = nn.Linear(de_hidden, de_hidden)
        self.de_relu2 = nn.ReLU()
        self.de_linear3 = nn.Linear(de_hidden, output_size)
        self.de_relu3 = nn.ReLU()

    def forward_elbo(self, x):
        """Estimate the ELBO for the mini-batch of data

        Parameters
        ----------
        x:  torch.Tensor
             Mini-batch of the observation data
        Returns
        -------
        reconstruction :
            reconstructed data
        mu:
            Means of the q_i distributions
        sigma:
            standard deviations of the q_i distributions

        """
        mu, sigma = self.encoder(x)
        z = self.sample_reparameterize(mu, sigma)
        reconstruction = self.decoder(z)
        return reconstruction, mu, sigma

    def loss_function(self, x, reconstruction, mu, sigma):
        """function that calculate loss

        Parameters
        ----------
        x:  torch.Tensor
             Mini-batch of the observation data
        reconstruction :
            reconstructed data
        mu:
            Means of the q_i distributions
        sigma:
            standard deviations of the q_i distributions
        Returns
        -------
        outp:
            loss
        """
        mse_loss = nn.MSELoss(reduction='sum')
        kl = self.kl_divergence(mu, sigma)
        outp = -0.5 * mse_loss(x, reconstruction) - kl
        return outp

    def encoder(self, x):
        """function that calculate loss

        Parameters
        ----------
        x:  torch.Tensor
             Mini-batch of the observation data
        Returns
        -------
        mu:
            Means of the q_i distributions
        sigma:
            standard deviations of the q_i distributions
        """
        x = self.en_linear1(x)
        x = self.en_relu1(x)
        x = self.en_linear2(x)
        x = self.en_relu2(x)
        mu = self.en_linear3_mu(x)
        sigma = torch.exp(self.en_linear3_sigma(x))
        return mu, sigma

    def sample_reparameterize(self, mu, sigma):
        """function that calculate loss

        Parameters
        ----------
        mu:
            Means of the q_i distributions
        sigma:
            standard deviations of the q_i distributions
        Returns
        ----------
        Latent variables samples from q(z)
        """
        epsilon = torch.empty_like(mu).normal_(0., 1.)
        return epsilon * sigma + mu

    def decoder(self, z):
        """function that convert sampled latent variables z into observations x

        Parameters
        ----------
        z:
            Sampled latent variables
        Returns
        -------
        Parameters of the conditional likelihood p(x/z)
        """
        z = self.de_linear1(z)
        z = self.de_relu1(z)
        z = self.de_linear2(z)
        z = self.de_relu2(z)
        z = self.de_linear3(z)
        z = self.de_relu3(z)
        return z

    def kl_divergence(self, mu, sigma):
        """function that calculate kl divergence

        Parameters
        ----------
        mu:
            Means of the q_i distributions
        sigma:
            standard deviations of the q_i distributions
        Returns
        ----------
        kl divergence for each of the q_i distributions
        """
        return 0.5 * (mu.pow(2) + sigma.pow(2) - 2 * torch.log(sigma) - 1).sum(-1)

    def gen_sample_data(self, sample_data_size, latent_dim):
        """function that calculate kl divergence

        Parameters
        ----------
        sample_data_size: int
            number of required sample
        latent_dim: int
            dimension of latent space

        Returns
        ----------
        gendata:
            generated sample data
        """
        distr = torch.distributions.normal.Normal(0, 1)
        z = distr.sample(torch.Size([sample_data_size, latent_dim]))
        gendata = self.decoder(z)
        return gendata


def show_mnist(nr_row, nr_col, data_loader):
    """function that show the mnist image

    Parameters
    ----------
    nr_row: int
        number of rows
    nr_col: int
        number of columns
    data_loader:  torch.utils.data.DataLoader
        dataloader
    """
    fig = plt.figure(figsize=(10, 6))
    nr_total = nr_row * nr_col
    for i in range(nr_total):
        idx = np.random.randint(0, len(data_loader))
        ax = fig.add_subplot(nr_row, nr_col, i + 1)
        image = data_loader.dataset[idx][0].view(28, 28).numpy()
        ax.imshow(image)
        ax.axis("off")
    plt.show()


class VAEmi(nn.Module):
    """Initialize the VAE model for MI building

    Parameters
    ----------
    en_hidden : list
        hidden dimension of encoder in different layers
    latent_space: int
        dimension of the observed data
    de_hidden: list
        hidden dimension of decoder in different layers
    input_size: int
        input dimension
    output_size: int
        output dimension
    """
    def __init__(self, en_hidden, latent_space, de_hidden, input_size, output_size):
        super().__init__()

        self.en_linear1 = nn.Linear(input_size, en_hidden[0])
        self.en_relu1 = nn.ReLU()
        self.en_linear2 = nn.Linear(en_hidden[0], en_hidden[1])
        self.en_relu2 = nn.ReLU()
        self.en_linear3_mu = nn.Linear(en_hidden[-1], latent_space)
        self.en_linear3_sigma = nn.Linear(en_hidden[-1], latent_space)
        self.de_linear1 = nn.Linear(latent_space, de_hidden[0])
        self.de_relu1 = nn.ReLU()
        self.de_linear2 = nn.Linear(de_hidden[0], de_hidden[1])
        self.de_relu2 = nn.ReLU()
        self.de_linear3 = nn.Linear(de_hidden[1], output_size)
        self.de_relu3 = nn.ReLU()

    def forward_elbo(self, x):
        """Estimate the ELBO for the mini-batch of data

        Parameters
        ----------
        x:  torch.Tensor
             Mini-batch of the observation data
        Returns
        -------
        reconstruction :
            reconstructed data
        mu:
            Means of the q_i distributions
        sigma:
            standard deviations of the q_i distributions

        """
        mu, sigma = self.encoder(x)
        z = self.sample_reparameterize(mu, sigma)
        reconstruction = self.decoder(z)
        return reconstruction, mu, sigma

    def loss_function(self, x, reconstruction, mu, sigma):
        """function that calculate loss

        Parameters
        ----------
        x:  torch.Tensor
             Mini-batch of the observation data
        reconstruction :
            reconstructed data
        mu:
            Means of the q_i distributions
        sigma:
            standard deviations of the q_i distributions
        Returns
        -------
        outp:
            loss
        """
        mse_loss = nn.MSELoss(reduction='sum')
        kl = self.kl_divergence(mu, sigma)
        outp = -0.5 * mse_loss(x, reconstruction) - kl
        return outp

    def encoder(self, x):
        """function that calculate loss

        Parameters
        ----------
        x:  torch.Tensor
             Mini-batch of the observation data
        Returns
        -------
        mu:
            Means of the q_i distributions
        sigma:
            standard deviations of the q_i distributions
        """
        x = self.en_linear1(x)
        x = self.en_relu1(x)
        x = self.en_linear2(x)
        x = self.en_relu2(x)
        mu = self.en_linear3_mu(x)
        sigma = torch.exp(self.en_linear3_sigma(x))
        return mu, sigma

    def sample_reparameterize(self, mu, sigma):
        """function that calculate loss

        Parameters
        ----------
        mu:
            Means of the q_i distributions
        sigma:
            standard deviations of the q_i distributions
        Returns
        ----------
        Latent variables samples from q(z)
        """
        epsilon = torch.empty_like(mu).normal_(0., 1.)
        return epsilon * sigma + mu

    def decoder(self, z):
        """function that convert sampled latent variables z into observations x

        Parameters
        ----------
        z:
            Sampled latent variables
        Returns
        -------
        Parameters of the conditional likelihood p(x/z)
        """
        z = self.de_linear1(z)
        z = self.de_relu1(z)
        z = self.de_linear2(z)
        z = self.de_relu2(z)
        z = self.de_linear3(z)
        z = self.de_relu3(z)
        return z

    def kl_divergence(self, mu, sigma):
        """function that calculate kl divergence

        Parameters
        ----------
        mu:
            Means of the q_i distributions
        sigma:
            standard deviations of the q_i distributions
        Returns
        ----------
        kl divergence for each of the q_i distributions
        """
        return 0.5 * (mu.pow(2) + sigma.pow(2) - 2 * torch.log(sigma) - 1).sum(-1)

    def gen_sample_data(self, sample_data_size, latent_dim):
        """function that calculate kl divergence

        Parameters
        ----------
        sample_data_size: int
            number of required sample
        latent_dim: int
            dimension of latent space

        Returns
        ----------
        gendata:
            generated sample data
        """
        distr = torch.distributions.normal.Normal(0, 1)
        z = distr.sample(torch.Size([sample_data_size, latent_dim]))
        gendata = self.decoder(z)
        return gendata


def show_mnist(nr_row, nr_col, data_loader):
    """function that show the mnist image

    Parameters
    ----------
    nr_row: int
        number of rows
    nr_col: int
        number of columns
    data_loader:  torch.utils.data.DataLoader
        dataloader
    """
    fig = plt.figure(figsize=(10, 6))
    nr_total = nr_row * nr_col
    for i in range(nr_total):
        idx = np.random.randint(0, len(data_loader))
        ax = fig.add_subplot(nr_row, nr_col, i + 1)
        image = data_loader.dataset[idx][0].view(28, 28).numpy()
        ax.imshow(image)
        ax.axis("off")
    plt.show()


def show_reconstruct_mnist(nr_row, nr_col, reconstr, labels):
    """function that show the mnist image

    Parameters
    ----------
    nr_row: int
        number of rows
    nr_col: int
        number of columns
    reconstr:
        reconstructed minist data
    labels:
        ground truth labels
    """
    fig = plt.figure(figsize=(10, 6))
    nr_total = nr_row * nr_col
    for i in range(nr_total):
        idx = np.random.randint(0, len(reconstr))
        ax = fig.add_subplot(nr_row, nr_col, i + 1)
        image = reconstr[idx].view(28, 28).detach().numpy()
        ax.imshow(image)
        ax.axis("off")
        ax.set_title("ground truth {0}".format(labels[idx].item()))
    plt.show()


def show_generate_mnist(nr_row, nr_col, gendata):
    """function that show the mnist image

    Parameters
    ----------
    nr_row: int
        number of rows
    nr_col: int
        number of columns
    gendata:
        generated data
    """
    fig = plt.figure(figsize=(10, 6))
    nr_total = nr_row * nr_col
    for i in range(nr_total):
        idx = np.random.randint(0, len(gendata))
        ax = fig.add_subplot(nr_row, nr_col, i + 1)
        image = gendata[idx].view(28, 28).detach().numpy()
        ax.imshow(image)
        ax.axis("off")
    plt.show()

def show_latent_distribution(latent_data, labels):
    """function that show the mnist image

    Parameters
    ----------
    latent_data: Tensor
        latent data after encoder
    labels: int
        ground truth labels
    """
    latent_z = latent_data.detach().numpy()
    plt.scatter(latent_z[:, 0], latent_z[:, 1], c=labels, cmap='tab10')
    plt.show()


if __name__ == "__main__":
    pass
        
       

