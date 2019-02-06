import argparse

defaults = {
	"dataset": True,
	"binarize_input": True,
	"epochs": 10,
	"loss": 'CE',
	"learning_rate": 3e-4,
	"latent_dim": 10,
	"print_every": 1
}

def check_args(args):
	"""Check commandline argument validity."""
	assert args.dataset=='MNIST' or args.dataset=='CIFAR10', "Dataset must be either 'MNIST' for 'CIFAR10"
	
	assert args.epochs >= 1, "Number of epochs must be a positive integer"
	
	assert args.learning_rate > 0, "Learning rate must be positive"
	
	assert args.latent_dim >= 1, "Latent dimension must be a positive integer"
	
	assert args.print_every >= 1, "print_every must be a positive integer"
	
	return args

def get_args():
	"""Parse arguments from commandline."""
	parser = argparse.ArgumentParser(
		description="Pytorch Implementation of Denoising Autoencoder(DAE)")
	
	parser.add_argument("-d", "--dataset",
		type=str, default=defaults['dataset'], help="'MNIST' or 'CIFAR10'")
	
	parser.add_argument("-e", "--epochs",
		type=int, default=defaults['epochs'], help="Number of epochs to train")
	
	parser.add_argument("-lr", "--learning_rate",
		type=float, default=defaults['learning_rate'], help="Learning rate for adam optimizer")
	
	parser.add_argument("-z", "--latent_dim",
		type=int, default=defaults['latent_dim'], help="Dimension of latent variable z")
	
	parser.add_argument("-p", "--print_every",
		type=int, default=defaults['print_every'], help="How often to print loss progress")
	
	return check_args(parser.parse_args())