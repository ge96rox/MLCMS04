import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt



class VAE(nn.Module):
    def __init__(self, en_hidden, latent_space, de_hidden, input_size, output_size):
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
        mu, sigma = self.encoder(x)
        z = self.sample_reparameterize(mu, sigma)
        reconstruction = self.decoder(z)
        return reconstruction, mu, sigma

    def loss_function(self, x, reconstruction, mu, sigma):
        mse_loss = nn.MSELoss(reduction='sum')
        kl = self.kl_divergence(mu, sigma)
        outp = -0.5 * mse_loss(x, reconstruction) - kl
        return outp

    def encoder(self, x):
        x = self.en_linear1(x)
        x = self.en_relu1(x)
        x = self.en_linear2(x)
        x = self.en_relu2(x)
        mu = self.en_linear3_mu(x)
        sigma = torch.exp(self.en_linear3_sigma(x))
        return mu, sigma

    def sample_reparameterize(self, mu, sigma):
        epsilon = torch.empty_like(mu).normal_(0., 1.)
        return epsilon * sigma + mu

    def decoder(self, z):
        z = self.de_linear1(z)
        z = self.de_relu1(z)
        z = self.de_linear2(z)
        z = self.de_relu2(z)
        z = self.de_linear3(z)
        z = self.de_relu3(z)
        return z

    def kl_divergence(self, mu, sigma):
        return 0.5 * (mu.pow(2) + sigma.pow(2) - 2 * torch.log(sigma) - 1).sum(-1)

    def gen_sample_data(self, sample_data_size, latent_dim):
        distr = torch.distributions.normal.Normal(0, 1)
        z = distr.sample(torch.Size([sample_data_size, latent_dim]))
        gendata = self.decoder(z)
        return gendata


def show_mnist(nr_row, nr_col, data_loader):
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
    def __init__(self, en_hidden, latent_space, de_hidden, input_size, output_size):
        super().__init__()

        self.en_linear1 = nn.Linear(input_size, en_hidden[0])
        #self.en_bn1 = nn.BatchNorm1d(en_hidden)
        self.en_relu1 = nn.ReLU()
        #self.en_dropout = nn.Dropout(p=0.5)
        self.en_linear2 = nn.Linear(en_hidden[0], en_hidden[1])
        #self.en_bn2 = nn.BatchNorm1d(en_hidden)
        self.en_relu2 = nn.ReLU()
        self.en_linear3_mu = nn.Linear(en_hidden[-1], latent_space)
        self.en_linear3_sigma = nn.Linear(en_hidden[-1], latent_space)
        self.de_linear1 = nn.Linear(latent_space, de_hidden[0])
        #self.de_bn1 = nn.BatchNorm1d(de_hidden)
        self.de_relu1 = nn.ReLU()
        #self.de_dropout = nn.Dropout(p=0.5)
        self.de_linear2 = nn.Linear(de_hidden[0], de_hidden[1])
        #self.de_bn2 = nn.BatchNorm1d(en_hidden)
        self.de_relu2 = nn.ReLU()
        self.de_linear3 = nn.Linear(de_hidden[1], output_size)
        self.de_relu3 = nn.ReLU()

        

    def forward_elbo(self, x):
        mu, sigma = self.encoder(x)
        z = self.sample_reparameterize(mu, sigma)
        reconstruction = self.decoder(z)
        return reconstruction, mu, sigma

    def loss_function(self, x, reconstruction, mu, sigma):
        mse_loss = nn.MSELoss(reduction='sum')
        kl = self.kl_divergence(mu, sigma)
        outp = -0.5 * mse_loss(x, reconstruction) - kl
        return outp

    def encoder(self, x):
        x = self.en_linear1(x)
        #x = self.en_bn1(x)
        x = self.en_relu1(x)
        #x = self.en_dropout(x)
        x = self.en_linear2(x)
        #x = self.en_bn2(x)
        x = self.en_relu2(x)
        mu = self.en_linear3_mu(x)
        sigma = torch.exp(self.en_linear3_sigma(x))
        return mu, sigma

    def sample_reparameterize(self, mu, sigma):
        epsilon = torch.empty_like(mu).normal_(0., 1.)
        return epsilon * sigma + mu

    def decoder(self, z):
        z = self.de_linear1(z)
        #z = self.de_bn1(z)
        z = self.de_relu1(z)
        #z = self.de_dropout(z)
        z = self.de_linear2(z)
        #z = self.de_bn2(z)
        z = self.de_relu2(z)
        z = self.de_linear3(z)
        z = self.de_relu3(z)
        return z

    def kl_divergence(self, mu, sigma):
        return 0.5 * (mu.pow(2) + sigma.pow(2) - 2 * torch.log(sigma) - 1).sum(-1)

    def gen_sample_data(self, sample_data_size, latent_dim):
        distr = torch.distributions.normal.Normal(0, 1)
        z = distr.sample(torch.Size([sample_data_size, latent_dim]))
        gendata = self.decoder(z)
        return gendata


def show_mnist(nr_row, nr_col, data_loader):
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
    latent_z = latent_data.detach().numpy()
    plt.scatter(latent_z[:, 0], latent_z[:, 1], c=labels, cmap='tab10')
    plt.show()


if __name__ == "__main__":
    test_set = torchvision.datasets.MNIST('dataset', train=False, download=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          )
    test_size = len(test_set)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_size, shuffle=True)

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('dataset', train=True, download=True,
                                   transform=torchvision.transforms.ToTensor(),
                                   ), batch_size=batch_size, shuffle=True)

    max_epochs = 101
    show_step = 100
    input_size = 28 * 28
    output_size = 28 * 28
    latent_dim = 2
    en_hidden = 256
    de_hidden = 256
    gendata_size = 100
    vae = VAE(en_hidden, latent_dim, de_hidden, input_size, output_size)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3, weight_dacay=1e-5)

    ###
    scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode='min', 
                patience=20, verbose=True, factor=0.5, min_lr=1e-7)
    ###

    loss_train = []
    loss_test = []
    loss_epoch = 0
    loss_test_epoch = 0
    i = 0
    for epoch in range(max_epochs):
        print("Epoch {0}".format(epoch))
        for batch in train_loader:
            x, y = batch
            x = x.view(x.shape[0], -1)
            opt.zero_grad()
            reconstruction, mu, sigma = vae.forward_elbo(x)
            loss = 1/batch_size*-vae.loss_function(x, reconstruction, mu, sigma).mean(-1)
            loss.backward()
            opt.step()
            loss_epoch += loss
            i += 1

        loss_epoch = loss_epoch / i
        print("loss = {0}".format(loss_epoch.item()))
        loss_train.append(loss_epoch)

        #####
        scheduler.step(loss_test)
        #####

        loss_epoch = 0
        i = 0

        
        for batch_test in test_loader:
            x_test, y_test = batch_test
            x_test = x_test.view(x_test.shape[0], -1)
            with torch.no_grad():
                reconstruction_test, mu_test, sigma_test = vae.forward_elbo(x_test)
                loss = 1/test_size*-vae.loss_function(x_test, reconstruction_test, mu_test, sigma_test).mean(-1)
                loss_test.append(loss)

            if epoch == 1 or epoch == 5 or epoch == 25 or epoch == 50 or epoch == 75 or epoch == 100:
               print("latent parameters")
               show_latent_distribution(mu_test,y_test)
               print("reconstruct images")
               show_reconstruct_mnist(3, 5, reconstruction_test, y_test)
               print("generate synthetic data")
               gen_data = vae.gen_sample_data(gendata_size, latent_dim)
               show_generate_mnist(3, 5, gen_data)
        
       

