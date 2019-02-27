import torchvision
import numpy as np
import matplotlib.pyplot as plt

from constants import *


def display_batch(batch, title, save=True):
    """Display batch of image using plt"""
    im = torchvision.utils.make_grid(batch, nrow=int(batch.shape[0]**0.5), padding=2, normalize=(DATASET=='CelebA'))
    plt.title(title)
    plt.imshow(np.transpose(im.cpu().detach().numpy(), (1, 2, 0)))
    if save:
        plt.savefig('results/' + title + '.png', transparent=True, bbox_inches='tight')
    plt.show()

def display_loss(D_loss, G_loss, save=True):
    """Display change of loss using plt"""
    plt.figure(figsize=(10,5))
    plt.title("Discriminator and Generator loss")
    plt.plot(G_loss, label='Generator')
    plt.plot(D_loss, label='Discriminator')
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    if save:
        plt.savefig(f'results/loss_{DATASET}.png')
    plt.show()