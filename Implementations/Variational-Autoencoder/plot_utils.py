import torchvision
import numpy as np
import matplotlib.pyplot as plt

def display_batch(title, batch, data, save):
    """Display and save batch of image using plt"""
    im = torchvision.utils.make_grid(batch)
    plt.title(title)
    plt.imshow(np.transpose(im.cpu().numpy(), (1, 2, 0)), cmap='gray')
    if save:
        plt.savefig('/results/'+title+data)
    plt.show()