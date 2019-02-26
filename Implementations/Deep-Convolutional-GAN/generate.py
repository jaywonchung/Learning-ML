import sys

import torch
import torch.nn as nn

from constants import *
from plot_utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() and NGPU > 0 else 'cpu')

def main(D_root='saved_model/discriminator_e19', G_root='saved_model/generator_e19'):

    # Load trained model
    D = torch.load(D_root, map_location=device)
    G = torch.load(G_root, map_location=device)

    # Peel nn.DataParallel if necessary
    if device == torch.device('cpu'):
        if isinstance(D, nn.DataParallel):
            D = D.module
        if isinstance(G, nn.DataParallel):
            G = G.module
    
    # Generate images
    noise = torch.randn(128, 100, 1, 1)
    image_batch = G(noise)[:64, :, :, :]
    display_batch(image_batch, 'random-generation')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
