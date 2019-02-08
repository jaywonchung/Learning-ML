import sys

import torch
import torchvision
import matplotlib.pyplot as plt

from plot_utils import display_and_save_batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_uniformly(num=81, grid_size=0.1, PATH=None, model=None):
    """
    Generates imgaes with 2D latent variables sampled uniformly with mean 0
    Currently supports Bernoulli decoders and Gaussian decoders without sigmas

    Args:
        num: Number of samples to make. Accepts (2*n+1)^2 numbers
        grid_size: Distance between adjacent latent variables
        PATH: The path to saved model (saved with torch.save(model, path))
        model: The trained model itself
    
    Note:
        Specify only one of PATH or model, not both
    """

    # Check arguments
    if num!=(int(num**0.5))**2 or (int(num**0.5))%2==0:
        raise ValueError('Argument num should be the square of an odd number')
    if PATH and model:
        raise ValueError('Pass either PATH or model, but not both')
    elif PATH is None and model is None:
        raise ValueError('You passed neither PATH nor model')
    
    # Load model and send to device
    if PATH:
        model = torch.load(PATH)
    model = model.to(device)

    # Sample tensor of latent variables
    side = num**0.5
    axis = (torch.arange(side) - side//2) * grid_size
    x = axis.reshape(1, -1)
    y = x_axis.transpose(0, 1)
    z = torch.stack(torch.broadcast_tensors(x, y), 0).to(device)

    # Generate output from decoder
    output = model.decoder(z)
    if PATH is None:
        PATH = f'{model.dataset}-{model.decoder_type}-z{model.latent_dim}'
    display_and_save_batch('Uniform generation', output, '-'+PATH)
    
if __name__=="__main__":
    generate_uniformly(PATH=sys.argv[0])