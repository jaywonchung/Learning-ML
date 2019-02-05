import sys

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from model import Autoencoder
from plot_utils import display_batch
from train_args import get_args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def CE_criterion(x, y):
	"""
	Cross-Entropy loss function.
	
	Args:
		x: output of autoencoder
		y: input data
	"""
	x = torch.clamp(x, 1e-8, 1-1e-8)
	return -torch.mean(y * torch.log(x) + (1.-y) * torch.log(1.-x))
	
def MSE_criterion(x, y):
	"""
	Mean Square Error loss function.
	
	Args:
		x: output of autoencoder
		y: input data
	"""
	return torch.mean((x-y)**2)

def main(args):
	"""
	Main function that trains the model.
	1. Retrieve arguments from args
	2. Prepare MNIST
	3. Train
	4. Display first batch of test set
	
	Args:
		args.add_noise: Whether to add noise (DAE) to input image or not (AE).not
		args.epochs: How many epochs to train model.
		args.loss: Which loss function to use. Either cross-entropy or mean square error.
		args.lr: Learning rate.
		args.latent_dim: Dimension of latent variable.
		args.print_every: How often to print training progress.
	"""
	# Retrieve arguments
	add_noise = args.add_noise
	epochs = args.epochs
	loss = args.loss
	lr = args.learning_rate
	latent_dim = args.latent_dim
	print_every = args.print_every
	
	# Load and transform MNIST dataset
	trsf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x : (x >= 0.5).float())])
	MNIST_train = datasets.MNIST(
		root='MNIST', train=True, transform=trsf, download=True)
	MNIST_test = datasets.MNIST(
		root='MNIST', train=False, transform=trsf, download=True)
	
	# Create dataloader
	train_loader = torch.utils.data.DataLoader(MNIST_train, batch_size=64, shuffle=True)
	test_loader = torch.utils.data.DataLoader(MNIST_test, batch_size=64, shuffle=False)
	
	# Create model and optimizer
	autoencoder = Autoencoder(latent_dim=latent_dim).to(device)
	optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
	
	# Select loss function
	criterion = CE_criterion if loss=='CE' else MSE_criterion
	
	# Train
	autoencoder.train()
	for epoch in range(epochs):
		for batch_ind, (input_data, _) in enumerate(train_loader):
			input_data = input_data.to(device)
			
			# Forward propagation with noise added
			if add_noise:
				output = autoencoder(F.dropout(input_data, p=0.5))
			else:
				output = autoencoder(input_data)
			
			# Calculate loss
			loss = criterion(output, input_data)
			
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

	# Display training result
	with torch.no_grad():
		images, _ = iter(test_loader).next()
		images = images.to(device)
		
		if add_noise:
			noise_images = F.dropout(images, p=0.5)
			output = autoencoder(noise_images)
			display_batch(images)
			display_batch(noise_images)
			display_batch(output)
		else:
			output = autoencoder(images)
			display_batch(images)
			display_batch(output)

if __name__ == "__main__":
	args = get_args()
	main(args)