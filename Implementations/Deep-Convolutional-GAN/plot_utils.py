import torchvision
import numpy as np
import matplotlib.pyplot as plt


def display_batch(batch, title, save=True):
    """Display batch of image using plt"""
    im = torchvision.utils.make_grid(batch, nrow=int(batch.shape[0]**0.5), padding=2, normalize=True)
    plt.title(title)
    plt.imshow(np.transpose(im.cpu().detach().numpy(), (1, 2, 0)))
    if save:
        plt.savefig('results/' + title + '.png', transparent=True, bbox_inches='tight')
    plt.show()
