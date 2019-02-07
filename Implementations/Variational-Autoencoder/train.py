import sys

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

from model import VAE
from plot_utils import display_batch
from arguments import get_args, defaults

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(**kwargs):
	"""
	Main function that trains the model
	1. Retrieve arguments from kwargs
	2. Prepare data
	3. Train
	4. Display first batch of test set
	
	Args:
        dataset: Which dataset to use
        decoder_type: How to model the output pixels, Gaussian or Bernoulli
        model_sigma: In case of Gaussian decoder, whether to model the sigmas too
		epochs: How many epochs to train model
        batch_size: Size of training / testing batch
		lr: Learning rate
		latent_dim: Dimension of latent variable
		print_every: How often to print training progress
	"""
	# Retrieve arguments
	dataset = kwargs.get('dataset', defaults['dataset'])
    decoder_type = kwargs.get('decoder_type', defaults['decoder_type'])
    model_sigma = kwargs.get('model_sigma', defaults['model_sigma'])
	epochs = kwargs.get('epochs', defaults['epochs'])
	batch_size = kwargs.get('batch_size', defaults['batch_size'])
	lr = kwargs.get('learning_rate', defaults['learning_rate'])
	latent_dim = kwargs.get('latent_dim', defaults['latent_dim'])
	print_every = kwargs.get('print_every', defaults['print_every'])
	
	# Specify dataset transform on load
    if decoder_type == 'Bernoulli':
        trsf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x : (x >= 0.5).float())])
    elif decoder_type == 'Gaussian':
        trsf = transforms.ToTensor()

    # Load dataset with transform
    if dataset == 'MNIST':
        train_data = datasets.MNIST(
            root='MNIST', train=True, transform=trsf, download=True)
        test_data = datasets.MNIST(
            root='MNIST', train=False, transform=trsf, download=True)
    elif dataset == 'CIFAR10':
        train_data = datasets.CIFAR10(
            root='CIFAR10', train=True, transform=trsf, download=True)
        test_data = datasets.CIFAR10(
            root='CIFAR10', train=False, transform=trsf, download=True)
	
	# Create dataloader
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
	
	# Create model and optimizer
    model = VAE(latent_dim, dataset, decoder_type).to(device)
	optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
	
	# Train
	autoencoder.train()
	for epoch in range(epochs):
		for batch_ind, (input_data, _) in enumerate(train_loader):
			input_data = input_data.to(device)
			
			# Forward propagation
            if decoder_type == 'Bernoulli':
			    z_mu, z_sigma, p = model(input_data)
            elif model_sigma:
                z_mu, z_sigma, out_mu, out_sigma = model(input_data)
            else:
                z_mu, z_sigma, out_mu = model(input_data)

			# Calculate loss
            KL_divergence_i = 0.5 * torch.sum(z_mu**2 + z_sigma**2 - torch.log(1e-8+z_sigma**2) - 1., dim=1)
            if decoder_type == 'Bernoulli':
                reconstruction_loss_i = torch.sum(input_data*torch.log(p) + (1.-input_data)*torch.log(1.-p), dim=(1,2,3))
            elif model_sigma:
                reconstruction_loss_i = -0.5 * torch.sum(torch.log(out_sigma**2) + ((input_data-out_mu)**2)/(out_sigma**2), dim=(1,2,3))
            else:
                reconstruction_loss_i = -0.5 * torch.sum((input_data-out_mu)**2, dim=(1,2,3))
            ELBO_i = -reconstruction_loss_i + KL_divergence_i
            loss = -torch.mean(ELBO_i)
			
			# Backward propagation
			optimizer.zero_grad()
			loss.backward()
			
			# Update parameters
			optimizer.step()
			
			# Print progress
			if batch_ind % print_every == 0:
				train_log = 'Epoch {:2d}/{:2d}\tLoss: {:.6f}\tTrain: [{}/{} ({:.0f}%)]      '.format(
				    epoch+1, epochs, loss.cpu().item(), batch_ind+1, len(train_loader),
				                100. * batch_ind / len(train_loader))
				print(train_log, end='\r')
				sys.stdout.flush()

	# Display training result with test set
	with torch.no_grad():
		images, _ = iter(test_loader).next()
		images = images.to(device)
		
		if add_noise:
			noise_images = F.dropout(images, p=0.5)
			denoised_output = autoencoder(noise_images)
			output = autoencoder(images)
			display_batch("Binarized truth" if binarize_input else "Truth",
				images, binarize_input)
			display_batch("Truth with noise", noise_images, binarize_input)
			display_batch("Reconstruction of noised image", output, binarize_input)
			display_batch("Reconstruction of clean image", denoised_output, binarize_input)
		else:
			output = autoencoder(images)
			display_batch("Binarized truth" if binarize_input else "Truth",
				images, binarize_input)
			display_batch("Reconstruction", output, binarize_input)

if __name__ == "__main__":
	args = get_args()
	main(**vars(args))