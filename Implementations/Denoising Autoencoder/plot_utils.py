import torchvision
import numpy as np
import matplotlib.pyplot as plt

def display_batch(title, batch, binarize):
	"""Display batch of image using plt"""
	if binarize:
		batch[batch>=0.5] = 1.
		batch[batch<0.5] = 0.
	im = torchvision.utils.make_grid(batch)
	
	plt.title(title)
	plt.imshow(np.transpose(im.cpu().numpy(), (1, 2, 0)), cmap='gray')
	plt.show()