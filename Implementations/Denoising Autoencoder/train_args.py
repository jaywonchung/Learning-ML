import argparse


def check_args(args):
	"""Check commandline argument validity."""
	assert args.add_noise==True or args.add_noise==False, "Add noise must be either True or False"
	
	assert args.epochs >= 1, "Number of epochs must be a positive integer"
	
	assert args.loss=="CE" or args.loss=="MSE", "Loss function must be either CE or MSE"
	
	assert args.learning_rate > 0, "Learning rate must be positive"
	
	assert args.latent_dim >= 1, "Latent dimension must be a positive integer"
	
	assert args.print_every >= 1, "print_every must be a positive integer"
	
	return args

def get_args():
	"""Parse arguments from commandline."""
	parser = argparse.ArgumentParser(
		description="Pytorch Implementation of Denoising Autoencoder(DAE)")
	
	parser.add_argument("-n", "--add_noise",
		type=bool, default=True, help="model = DAE if add_noise else AE")
	
	parser.add_argument("-e", "--epochs",
		type=int, default=10, help="Number of epochs to train")
	
	parser.add_argument("-ls", "--loss",
		type=str, default="CE", help="Which loss function to use: CE or MSE")
	
	parser.add_argument("-lr", "--learning_rate",
		type=float, default=1e-3, help="Learning rate for adam optimizer")
	
	parser.add_argument("-z", "--latent_dim",
		type=int, default=10, help="Dimension of latent variable z")
	
	parser.add_argument("-p", "--print_every",
		type=int, default=1, help="How often to print loss progress")
	
	return check_args(parser.parse_args())