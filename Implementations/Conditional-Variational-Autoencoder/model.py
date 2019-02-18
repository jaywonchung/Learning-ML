import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GaussianEncoder(nn.Module):
    """Gaussian encoder module for VAE"""

    def __init__(self, latent_dim, dataset):
        """
        Constructor for the GaussianEncoder class

        Parameter:
            latent_dim: Dimension of the latent variable
            dataset: Type of dataset to use. Either 'MNIST' or 'CIFAR10'
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.dataset = dataset

        if dataset == 'MNIST':
            # x: (N, 1, 28, 38)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1)
            # x: (N, 64, 14, 19)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
            self.bn2 = nn.BatchNorm2d(num_features=128)
            # x: (N, 128, 7, 9)
            self.fc3 = nn.Linear(in_features=128*7*9, out_features=1024)
            self.bn3 = nn.BatchNorm1d(num_features=1024)
            # x: (N, 1024)
            self.fc4 = nn.Linear(in_features=1024, out_features=2*latent_dim)
            # x: (N, 2*latent_dim)

        elif dataset == 'CIFAR10':
            # x: (N, 3, 32, 32)
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)
            # x: (N, 64, 16, 16)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
            self.bn2 = nn.BatchNorm2d(num_features=128)
            # x: (N, 128, 8, 8)
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
            self.bn3 = nn.BatchNorm2d(num_features=256)
            # x: (N, 256, 4, 4)
            self.fc4 = nn.Linear(in_features=256*4*4, out_features=2*latent_dim)
            # x: (N, 2*latent_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.)

    def forward(self, x):
        """
        Forward method for the GaussianEncoder class

        Parameter:
            x: Batch of images. For MNIST, (N, 1, 28, 28). For CIFAR10, (N, 3, 32, 32).

        Return:
            mu: Vector of size latent_dim.
                Each element represents the mean of a Gaussian Distribution.
            sigma: Vector of size latent_dim.
                   Each element represents the standard deviation of a Gaussian distribution.
        """
        if self.dataset == 'MNIST':
            x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
            x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
            x = x.view(-1, 128*7*7)
            x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.2)
            x = self.fc4(x)

        elif self.dataset == 'CIFAR10':
            x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
            x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
            x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
            x = x.view(-1, 256*4*4)
            x = self.fc4(x)

        # split x in half
        mu = x[:, :self.latent_dim]
        # sigma shouldn't be negative
        sigma = 1e-6 + F.softplus(x[:, self.latent_dim:])

        return mu, sigma


class BernoulliDecoder(nn.Module):
    """BernoulliDecoder module for VAE with MNIST dataset"""

    def __init__(self, latent_dim):
        """
        Constructor for the BernoulliDecoder class
        Can only model black and white MNIST images

        Parameter:
            latent_dim: Dimension of the latent variable
        """
        super().__init__()

        # z: (N, latent_dim)
        self.fc1 = nn.Linear(in_features=latent_dim+10, out_features=1024)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        # z: (N, 1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=128*7*7)
        self.bn2 = nn.BatchNorm1d(num_features=128*7*7)
        # z: (N, 128*7*7)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        # z: (N, 64, 14, 14)
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1)
        # z: (N, 1, 28, 28)
        
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.)
    
    def forward(self, z):
        """
        Forward method for the BernoulliDecoder class

        Parameter:
            z: Batch of latent variables.

        Return:
            mu: Vector of size latent_dim.
                Each element represents the mean of a Gaussian Distribution.
            sigma: Vector of size latent_dim.
                   Each element represents the standard deviation of a Gaussian distribution.
        """
        z = F.relu(self.bn1(self.fc1(z)))
        z = F.relu(self.bn2(self.fc2(z)))
        z = z.view(-1, 128, 7, 7)
        z = F.relu(self.bn3(self.deconv3(z)))
        z = torch.sigmoid(self.deconv4(z))
        return z,


class GaussianDecoder(nn.Module):
    """GaussianDecoder module for VAE"""

    def __init__(self, latent_dim, dataset, model_sigma=False):
        """
        Constructor for the GaussianDecoder class

        Parameter:
            latent_dim: Dimension of the latent variable
            dataset: Type of dataset to use. Either 'MNIST' or 'CIFAR10'
            model_sigma: Whether to model standard deviations too.
                         If False, only outputs the mu vector, and all sigma is implicitly 1.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.dataset = dataset
        self.model_sigma = model_sigma

        if dataset == 'MNIST':
            # z: (N, latent_dim)
            self.fc1 = nn.Linear(in_features=latent_dim+10, out_features=1024)
            self.bn1 = nn.BatchNorm1d(num_features=1024)
            # z: (N, 1024)
            self.fc2 = nn.Linear(in_features=1024, out_features=128*7*7)
            self.bn2 = nn.BatchNorm1d(num_features=128*7*7)
            # z: (N, 128*7*7)
            self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
            self.bn3 = nn.BatchNorm2d(num_features=64)
            # z: (N, 64, 14, 14)
            if model_sigma:
                self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=4, stride=2, padding=1)
                # z: (N, 2, 28, 28)
            else:
                self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1)
                # z: (N, 1, 28, 28)

        elif dataset == 'CIFAR10':
            # z: (N, latent_dim)
            self.fc1 = nn.Linear(in_features=latent_dim+10, out_features=448*2*2)
            self.bn1 = nn.BatchNorm1d(num_features=448*2*2)
            # z: (N, 448*2*2)
            self.deconv2 = nn.ConvTranspose2d(in_channels=448, out_channels=256, kernel_size=4, stride=2, padding=1)
            self.bn2 = nn.BatchNorm2d(num_features=256)
            # z: (N, 256, 4, 4)
            self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
            # z: (N, 128, 8, 8)
            self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
            # z: (N, 64, 16, 16)
            if model_sigma:
                self.deconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=6, kernel_size=4, stride=2, padding=1)
                # z: (N, 6, 32, 32)
            else:
                self.deconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1)
                # z: (N, 3, 32, 32)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.)

    def forward(self, z):
        """
        Forward method for the GaussianDecoder class

        Parameter:
            z: Batch of latent variables.

        Return:
            mu: Vector of size latent_dim.
                Each element represents the mean of a Gaussian Distribution.
            sigma: Vector of size latent_dim.
                   Each element represents the standard deviation of a Gaussian distribution.
        """
        if self.dataset == 'MNIST':
            z = F.relu(self.bn1(self.fc1(z)))
            z = F.relu(self.bn2(self.fc2(z)))
            z = z.view(-1, 128, 7, 7)
            z = F.relu(self.bn3(self.deconv3(z)))
            if self.model_sigma:
                gaussian = self.deconv4(z)
                mu = torch.sigmoid(gaussian[:, 0, :, :]).unsqueeze(1)
                sigma = 1e-6 + F.softplus(gaussian[:, 1, :, :]).unsqueeze(1)
                return mu, sigma
            else:
                mu = torch.sigmoid(self.deconv4(z))
                return mu,
        
        elif self.dataset == 'CIFAR10':
            z = F.relu(self.bn1(self.fc1(z)))
            z = z.view(-1, 448, 2, 2)
            z = F.relu(self.bn2(self.deconv2(z)))
            z = F.relu(self.deconv3(z))
            z = F.relu(self.deconv4(z))
            if self.model_sigma:
                gaussian = self.deconv5(z)
                mu = torch.sigmoid(gaussian[:, :3, :, :]).unsqueeze(1)
                sigma = 1e-6 + F.softplus(gaussian[:, 3:, :, :]).unsqueeze(1)
                return mu, sigma
            else:
                mu = torch.sigmoid(self.deconv5(z))
                return mu,




class CVAE(nn.Module):
    """Conditional Variational Autoencoder module that wraps one encoder and one decoder module."""
    
    def __init__(self, latent_dim, dataset, decoder_type, model_sigma=False):
        """
        Constructor for VAE class

        Parameter:
            latent_dim: Dimension of the latent variable
            dataset: Type of dataset to use. Either 'MNIST' or 'CIFAR10'
            decoder_type: Which type of decoder to use. Either 'Bernoulli' or 'Gaussian'
            model_sigma: Whether to model standard deviations too.
                         If True, forward method returns (mu, sigma). Else, returns only mu.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.dataset = dataset
        self.decoder_type = decoder_type
        self.model_sigma = model_sigma

        self.encoder = GaussianEncoder(latent_dim, dataset)
        if decoder_type == 'Bernoulli':
            self.decoder = BernoulliDecoder(latent_dim)
        elif decoder_type == 'Gaussian':
            self.decoder = GaussianDecoder(latent_dim, dataset, model_sigma)
    
    def forward(self, x, y):
        """
        Forward method for the GaussianEncoder class
        Samples latent variable z from the distribution calculated by the encoder,
        and feeds it to the decoder

        Parameter:
            x: Batch of images. For MNIST, (N, 1, 28, 28). For CIFAR10, (N, 3, 32, 32).
            y: Label of batch.

        Return:
            If model_sigma is True, returns (z_mu, z_sigma, mu, sigma)
            Else, returns (z_mu, z_sigma, mu)
        """
        # Create one-hot vector from labels
        y = y.view(x.shape[0], 1)
        onehot_y = torch.zeros((x.shape[0], 10), device=device, requires_grad=False)
        onehot_y.scatter_(1, y, 1)

        # Encode
        input_batch = torch.cat((x, onehot_y.view(x.shape[0], 1, 1, 10)*torch.ones(x.shape[0], x.shape[1], x.shape[2], 10)), dim=3)
        z_mu, z_sigma = self.encoder(input_batch)
        self.z = z_mu + z_sigma * torch.randn_like(z_mu, device=device)  # reparametrization trick

        # Decode
        latent = torch.cat((self.z, onehot_y), dim=1)
        param = self.decoder(latent)
        return (z_mu, z_sigma) + param