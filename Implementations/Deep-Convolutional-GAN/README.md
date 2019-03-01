
# Deep Convolutional GAN for MNIST, CIFAR10, and CelebA

This is the pytorch implementation of:
- Deep Convolutional GAN (DCGAN)

which was introduced in [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) by Radford et al.

# Network Structure

Refer to [```model.py```](/Implementations/Deep-Convolutional-GAN/model.py).

For MNIST and CIFAR10, 4 conv-bn layers used. For CelebA, 5 conv-bn layers used.

# Results

Training specs:
- Initial learning rate ```2e-4``` with ```torch.optim.Adam``` optimizer
- Batch size ```128```
- 
## MNIST Generation

Generated with **2-D** fixed noise every epoch:

<table align='center'>
<tr align='center'>
	<td> Epoch 1 </td>
    <td> Epoch 2  </td>
    <td> Epoch 5 </td>
</tr>
<tr align='center'>
	<td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-MNIST-z2-e01.png' height = '300px'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-MNIST-z2-e02.png' height = '300px'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-MNIST-z2-e05.png' height = '300px'>
</tr>
<tr align='center'>
	<td> Epoch 10 </td>
    <td> Epoch 15  </td>
    <td> Epoch 20 </td>
</tr>
<tr align='center'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-MNIST-z2-e10.png' height = '300px'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-MNIST-z2-e15.png' height = '300px'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-MNIST-z2-e20.png' height = '300px'>
</tr>
</table>

Generated with **100-D** fixed noise every epoch:

<table align='center'>
<tr align='center'>
	<td> Epoch 1 </td>
    <td> Epoch 2  </td>
    <td> Epoch 5 </td>
</tr>
<tr align='center'>
	<td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-MNIST-z100-e01.png' height = '300px'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-MNIST-z100-e02.png' height = '300px'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-MNIST-z100-e05.png' height = '300px'>
</tr>
<tr align='center'>
	<td> Epoch 10 </td>
    <td> Epoch 20  </td>
    <td> Epoch 25 </td>
</tr>
<tr align='center'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-MNIST-z100-e10.png' height = '300px'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-MNIST-z100-e20.png' height = '300px'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-MNIST-z100-e25.png' height = '300px'>
</tr>
</table>

## CIFAR10 Generation

Generated with **100-D** fixed noise every epoch:

<table align='center'>
<tr align='center'>
    <td> Epoch 1 </td>
    <td> Epoch 10 </td>
    <td> Epoch 20 </td>
</tr>
<tr align='center'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-CIFAR10-z100-e01.png' height = '300px'> </td>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-CIFAR10-z100-e10.png' height = '300px'> </td>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-CIFAR10-z100-e20.png' height = '300px'> </td>
</tr>
<tr align='center'>
	<td> Epoch 30 </td>
    <td> Epoch 40  </td>
    <td> Epoch 50 </td>
</tr>
<tr align='center'>
	<td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-CIFAR10-z100-e30.png' height = '300px'> </td>
	<td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-CIFAR10-z100-e40.png' height = '300px'> </td>
	<td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-CIFAR10-z100-e50.png' height = '300px'> </td>
</tr>
</table>

## CelebA Generation

Generated with **100-D** fixed noise every epoch:

<table align='center'>
<tr align='center'>
    <td> Epoch 1 </td>
    <td> Epoch 2 </td>
    <td> Epoch 5 </td>
</tr>
<tr align='center'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-CelebA-z100-e01.png' height = '300px'> </td>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-CelebA-z100-e02.png' height = '300px'> </td>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-CelebA-z100-e05.png' height = '300px'> </td>
</tr>
<tr align='center'>
	<td> Epoch 10 </td>
    <td> Epoch 15  </td>
    <td> Epoch 20 </td>
</tr>
<tr align='center'>
	<td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-CelebA-z100-e10.png' height = '300px'> </td>
	<td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-CelebA-z100-e15.png' height = '300px'> </td>
	<td><img src = '/Implementations/Deep-Convolutional-GAN/results/fixed-noise-CelebA-z100-e20.png' height = '300px'> </td>
</tr>
</table>

# Usage
## Prerequisites

1. Pytorch and torchvision
2. Packages: numpy, matplotlib

## Configuration

You can configure the dataset, training specs, the latent variable dimension, and more in [```constants.py```](/Implementations/Deep-Convolutional-GAN/constants.py).

## Execution

Command line:
```bash
python train.py
```

Jupyter notebook:
```python
from train import main
%matplotlib inline

main()
```


