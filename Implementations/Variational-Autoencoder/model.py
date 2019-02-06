import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianEncoder(nn.Module):
	"""Gaussian encoder module for VAE"""

	def __init__(self, latent_dim=2, dataset='MNIST'):
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
            # x: (N, 1, 28, 28)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1)
            # x: (N, 64, 14, 14)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
            self.bn2 = nn.BatchNorm2d(num_features=128)
            # x: (N, 128, 7, 7)
            self.fc3 = nn.Linear(in_features=128*7*7, out_features=1024)
            self.bn3 = nn.BatchNorm1d(num_features=1024)
            # x: (N, 1024)
            self.fc4 = nn.Linear(in_features=1024, out_features=2*latent_dim)
            # x: (N, 2*latent_dim)

        elif dataset = 'CIFAR10':
            # x: (N, 3, 32, 32)
		    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)
            # x: (N, 64, 16, 16)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
            self.bn2 = nn.BatchNorm2d(num_features=64)
            # x: (N, 128, 8, 8)
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
            self.bn3 = nn.BatchNorm2d(num_features=128)
            # x: (N, 256, 4, 4)
            self.fc4 = nn.Linear(in_features=256*4*4, out_features=2*latent_dim)
            # x: (N, 2*latent_dim)

        else:
            raise NotImplementedError

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
        
        else:
            raise AttributeError('Wrong value for self.dataset: ' + self.dataset)

        # split x in half
        mu = x[:, :self.latent_dim]
        # sigma shouldn't be negative
        sigma = 1e-6 + F.softplus(x[:, self.latent_dim:])

        return mu, sigma

class BernoulliDecoder(nn.Module):
	"""BernoulliDecoder module for VAE with MNIST dataset"""

	def __init__(self, latent_dim=2):
        """
        Constructor for the BernoulliDecoder class
        Can only model black and white MNIST images

        Parameter:
            latent_dim: Dimension of the latent variable
        """
		super().__init__()
		self.fc1 = nn.Linear(latent_dim, 500)
		self.fc2 = nn.Linear(500, 500)
		self.fc3 = nn.Linear(500, 28*28)
		
		for m in self.modules():
            if isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight)
				m.bias.data.fill_(0.)
	
	def forward(self, z):
        """
        Forward method for the BernoulliDecoder class

        Parameter:
            x: Batch of images. For MNIST, (N, 1, 28, 28). For CIFAR10, (N, 3, 32, 32).

        Return:
            mu: Vector of size latent_dim.
                Each element represents the mean of a Gaussian Distribution.
            sigma: Vector of size latent_dim.
                   Each element represents the standard deviation of a Gaussian distribution.
        """
		x = self.fc1(x)
		x = torch.tanh(x)
		x = F.dropout(x, p=0.1)
		
		x = self.fc2(x)
		x = F.elu(x)
		x = F.dropout(x, p=0.1)
		
		x = self.fc3(x)
		x = torch.sigmoid(x)
		
		x = x.view(-1, 1, 28, 28)
		return x

class GaussianDeccoder(nn.Module):
    """GaussianDecoder module for VAE"""

    def __init__(self, latent_dim=2, dataset='MNIST'):
        """
        Constructor for the GaussianDecoder class

        Parameter:
            latent_dim: Dimension of the latent variable
            dataset: Type of dataset to use. Either 'MNIST' or 'CIFAR10'
        """
        super().__init__()

    def forward(self, z):
        """
        Forward method for the GaussianDecoder class

        Parameter:
            x: Batch of images. For MNIST, (N, 1, 28, 28). For CIFAR10, (N, 3, 32, 32).

        Return:
            mu: Vector of size latent_dim.
                Each element represents the mean of a Gaussian Distribution.
            sigma: Vector of size latent_dim.
                   Each element represents the standard deviation of a Gaussian distribution.
        """
        

class VAE(nn.Module):
	"""Variational Autoencoder module that wraps one encoder and one decoder module."""
	
	def __init__(self, latent_dim=2, dataset='MNIST', decoder_type='Bernoulli'):
        """
        Constructor for VAE class

        Parameter:
            latent_dim: Dimension of the latent variable
            dataset: Type of dataset to use. Either 'MNIST' or 'CIFAR10'
            decoder_type: Which type of decoder to use. Either 'Bernoulli' or 'Gaussian'
        """
		super().__init__()
		self.encoder = Encoder(latent_dim)
		self.decoder = Decoder(latent_dim)
	
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
		return self.decoder(self.encoder(x))