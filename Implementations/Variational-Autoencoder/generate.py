import sys

import torch
import torchvision
import matplotlib.pyplot as plt

from plot_utils import display_and_save_batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_uniformly(num=400, grid_size=0.1, PATH=None, model=None):
    """
    Generates imgaes with 2D latent variables sampled uniformly with mean 0
    Currently supports Bernoulli decoders and Gaussian decoders without sigmas

    Args:
        num: Number of samples to make. Accepts square numbers
        grid_size: Distance between adjacent latent variables
        PATH: The path to saved model (saved with torch.save(model, path))
        model: The trained model itself
    
    Note:
        Specify only one of PATH or model, not both
    """

    # Check arguments
    if num!=(int(num**0.5))**2:
        raise ValueError('Argument num should be a square number')
    if PATH and model:
        raise ValueError('Pass either PATH or model, but not both')
    elif PATH is None and model is None:
        raise ValueError('You passed neither PATH nor model')
    
    # Load model and send to device
    if PATH:
        model = torch.load(PATH, map_location='cpu')
    model = model.to(device)

    # Sample tensor of latent variables
    side = num**0.5
    axis = (torch.arange(side) - side//2) * grid_size
    x = axis.reshape(1, -1)
    y = x.transpose(0, 1)
    z = torch.stack(torch.broadcast_tensors(x, y), 2).reshape(-1, 2).to(device)

    # Generate output from decoder
    with torch.no_grad():
        output, = model.decoder(z)
        if PATH is None:
            PATH = f'{model.dataset}-{model.decoder_type}-z{model.latent_dim}'
        display_and_save_batch('Uniform generation', output, '-')
    
if __name__=="__main__":
    generate_uniformly(PATH=sys.argv[1])