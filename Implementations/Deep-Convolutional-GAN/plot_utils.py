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


def display_and_save_latent(batch, label, data, isplay=True):
    """Display and save batch of 2-D latent variable using plt"""
    colors = ['black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'pink', 'violet', 'grey']
    z = batch.cpu().detach().numpy()
    l = label.cpu().numpy()

    plt.title('Latent variables')
    plt.scatter(z[:,0], z[:,1], c=l, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlim(-3, 3, )
    plt.ylim(-3, 3)
    if save:
        plt.savefig('results/latent-variable' + data + '.png', transparent=True, bbox_inches='tight')
    plt.show()