
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

## Execution

Command line:
```bash
python train.py -d 'MNIST' -t 'Bernoulli' -s False -e 10 -b 64 -lr 3e-4 -z 10 -p 1 -rp 'saved_model/path' -re 20
```

Jupyter notebook:
```python
from train import main
%matplotlib inline

main(dataset='MNIST', decoder_type='Bernoulli', model_sigma=False, epochs=10, batch_size=64, learning_rate=3e-4, latent_dim=10, print_every=1, resume_path='saved_model/path', resume_epoch=20)
```

## Arguments
Every argument is optional, and has a default value defined at ```arguments.py```.

- ```--dataset, -d```: 'MNIST' or 'CIFAR10'. *Default*: - ```'MNIST'```  
- ```--decoder_type, -t```: 'Bernoulli' or 'Gaussian'. *Default*: ```'Bernoulli'```
- ```--model_sigma, -s```: In case of Gaussian decoder, whether to model the standard deviation. *Default*: ```False```
- ```--epochs, -e```: Number of epochs to train. *Default*: ```10```
- ```--batch_size, -b```: Size of batch size at training/testing. *Default*: ```64```
- ```--learning_rate, -lr```: Learning rate. *Default*: ```3e-4```
- ```--latent_dim, -z```: Dimension of the latent variable. *Default*: ```10```
- ```--print_every, -p```: How often to print training progress. *Default*: ```1```
- ```--resume_path, -rp```: In case you want to resume training from a saved model, provide its path here.
- ```--resume_epoch, -re```: Number of epochs already trained for the saved model. *Default*: ```0```

