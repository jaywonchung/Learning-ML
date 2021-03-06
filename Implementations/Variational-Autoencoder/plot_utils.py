import torchvision
import numpy as np
import matplotlib.pyplot as plt

def display_and_save_batch(title, batch, data, save=True, display=True):
    """Display and save batch of image using plt"""
    im = torchvision.utils.make_grid(batch, nrow=int(batch.shape[0]**0.5))
    plt.title(title)
    plt.imshow(np.transpose(im.cpu().numpy(), (1, 2, 0)), cmap='gray')
    if save:
        plt.savefig('results/'+title+data+'.png', transparent=True, bbox_inches='tight')
    if display:
        plt.show()