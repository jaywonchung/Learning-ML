import sys

import torch
import torchvision
import matplotlib.pyplot as plt

from plot_utils import display_and_save_batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_images(mode='uniform', num=400, grid_size=0.05, PATH=None, model=None):
    """
    Generates imgaes with 2D latent variables sampled uniformly with mean 0
    Currently supports Bernoulli decoders and Gaussian decoders without sigmas

    Args:
        mode: 'uniform' or 'random'
        num: Number of samples to make. Accepts square numbers
        grid_size: Distance between adjacent latent variables
        PATH: The path to saved model (saved with torch.save(model, path))
        model: The trained model itself
    
    Note:
        Specify only one of PATH or model, not both
    """

    # Check arguments
    if mode!='uniform' and mode!='random':
        raise ValueError("Argument mode should either be 'uniform' or 'random'")
    if num!=(int(num**0.5))**2:
        raise ValueError('Argument num should be a square number')
    if PATH and model:
        raise ValueError('Pass either PATH or model, but not both')
    elif PATH is None and model is None:
        raise ValueError('You passed neither PATH nor model')
    
    # Load model
    if PATH:
        model = torch.load(PATH, map_location=device)

    # Sample tensor of latent variables
    if mode == 'uniform':
        side = num**0.5
        axis = (torch.arange(side) - side//2) * grid_size
        x = axis.reshape(1, -1)
        y = x.transpose(0, 1)
        z = torch.stack(torch.broadcast_tensors(x, y), 2).reshape(-1, 2).to(device)
    elif mode == 'random':
        z = torch.randn((num, 2), device=device)

    # Generate output from decoder
    with torch.no_grad():
        for i in range(10):
            label = torch.zeros(num, 10)
            label[:, i] = 1
            latent = torch.cat((z, label), dim=1)
            output, = model.decoder(latent)
            display_and_save_batch(f'{mode}-generation', output, f'-{model.dataset}-{num}-{i+1}')
    
if __name__=="__main__":
    # Commandline input
    generate_images(mode=sys.argv[1], PATH=sys.argv[2])