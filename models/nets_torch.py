import torch
from torch import nn
from torch.nn import functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def select_model(architecture):
    if architecture == 'linear':
        vae = VAELinear()
    elif architecture == 'nonlinear':
        vae = VAENonLinear()
    elif architecture == 'conv':
        vae = VAEConv()
    return vae


class VAELinear(nn.Module):
    def __init__(self):
        super(VAELinear, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 64)
        self.fc22 = nn.Linear(400, 64)
        self.fc3 = nn.Linear(64, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = self.fc1(x)
        mean = self.fc21(h1)
        std = self.fc22(h1)
        return mean, std

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        h3 = self.fc3(z)
        out = torch.sigmoid(self.fc4(h3))
        return out

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAENonLinear(nn.Module):
    def __init__(self):
        super(VAENonLinear, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 240)
        self.fc_mu = nn.Linear(240, 64)
        self.fc_logvar = nn.Linear(240, 64)
        self.fc3 = nn.Linear(64, 240)
        self.fc4 = nn.Linear(240, 400)
        self.fc5 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        mean = self.fc_mu(h2)
        logvar = self.fc_logvar(h2)
        return mean, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        x_hat = torch.sigmoid(self.fc5(h4))
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEConv(nn.Module):
    def __init__(self):
        super(VAEConv, self).__init__()

        self.l1 = nn.Conv2d(1, 32, 3, stride=(2, 2))
        self.l2 = nn.Conv2d(32, 64, 3, stride=(2, 2))
        self.l_mean = nn.Linear(6 * 6 * 64, 64)
        self.l_logvar = nn.Linear(6 * 6 * 64, 64)

        # self.l4 = nn.Linear(64, 6 * 6 * 64)
        # self.l5 = nn.ConvTranspose2d(64, 64, 3, stride=2)
        # self.l6 = nn.ConvTranspose2d(64, 32, 3, stride=2)
        # self.l7 = nn.ConvTranspose2d(32, 1, 2, stride=1)

        self.l4 = nn.Linear(64, 7 * 7 * 32)
        self.l5 = nn.ConvTranspose2d(32, 64, 2, stride=2)
        self.l6 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.l7 = nn.ConvTranspose2d(32, 1, 1, stride=1)

    def encode(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = torch.flatten(h2, start_dim=1)
        mean = self.l_mean(h3)
        logvar = self.l_logvar(h3)
        return mean, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        h1 = F.relu(self.l4(z))
        # h2 = h1.view(-1, 64, 6, 6)
        h2 = h1.view(-1, 32, 7, 7)
        h3 = F.relu(self.l5(h2))
        h4 = F.relu(self.l6(h3))
        x_hat = torch.sigmoid(self.l7(h4))
        x_hat = x_hat.view(-1, 784)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
