
# Deep Convolutional GAN for MNIST, CIFAR10, and CelebA

This is the pytorch implementation of:
- Deep Convolutional GAN (DCGAN)

which was introduced in [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) by Radford et al.

# Network Structure

Refer to [```model.py```](/Implementations/Deep-Convolutional-GAN/model.py).

For MNIST and CIFAR10, 4 conv-bn layers used. For CelebA, 5 conv-bn layers used.

# Results
## MNIST Generation

With **2-D uniformly sampled** latent variables:

<table align='center'>
<tr align='center'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/uniform-generation-MNIST-400-0.png' height = '350px'></td>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/uniform-generation-MNIST-400-1.png' height = '350px'></td>
</tr>
<tr align='center'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/uniform-generation-MNIST-400-2.png' height = '350px'></td>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/uniform-generation-MNIST-400-3.png' height = '350px'></td>
</tr>
<tr align='center'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/uniform-generation-MNIST-400-4.png' height = '350px'></td>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/uniform-generation-MNIST-400-5.png' height = '350px'></td>
</tr>
<tr align='center'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/uniform-generation-MNIST-400-6.png' height = '350px'></td>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/uniform-generation-MNIST-400-7.png' height = '350px'></td>
</tr>
<tr align='center'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/uniform-generation-MNIST-400-8.png' height = '350px'></td>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/uniform-generation-MNIST-400-9.png' height = '350px'></td>
</tr>
</table>

With **2-D randomly sampled** latent variables:

<table align='center'>
<tr align='center'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/random-generation-MNIST-400-0.png' height = '350px'></td>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/random-generation-MNIST-400-1.png' height = '350px'></td>
</tr>
<tr align='center'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/random-generation-MNIST-400-2.png' height = '350px'></td>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/random-generation-MNIST-400-3.png' height = '350px'></td>
</tr>
<tr align='center'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/random-generation-MNIST-400-4.png' height = '350px'></td>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/random-generation-MNIST-400-5.png' height = '350px'></td>
</tr>
<tr align='center'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/random-generation-MNIST-400-6.png' height = '350px'></td>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/random-generation-MNIST-400-7.png' height = '350px'></td>
</tr>
<tr align='center'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/random-generation-MNIST-400-8.png' height = '350px'></td>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/random-generation-MNIST-400-9.png' height = '350px'></td>
</tr>
</table>

## CIFAR10 Reconstruction

With **Gaussian decoder** and **50-D** latent variable:

<table align='center'>
<tr align='center'>
    <td> Input image </td>
</tr>
<tr align='center'>
    <td> <img src = "/Implementations/Deep-Convolutional-GAN/results/Truth-Gaussian.png" height = '200px'> </td>
</tr>
<tr align='center'>
    <td> Epoch 1 </td>
    <td> Epoch 5 </td>
    <td> Epoch 25 </td>
</tr>
<tr align='center'>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/Mean-reconstruction-Gaussian-z50-e001.png' height = '200px'> </td>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/Mean-reconstruction-Gaussian-z50-e005.png' height = '200px'> </td>
    <td><img src = '/Implementations/Deep-Convolutional-GAN/results/Mean-reconstruction-Gaussian-z50-e025.png' height = '200px'> </td>
</tr>
<tr align='center'>
	<td> Epoch 50 </td>
    <td> Epoch 75  </td>
    <td> Epoch 100 </td>
</tr>
<tr align='center'>
	<td><img src = '/Implementations/Deep-Convolutional-GAN/results/Mean-reconstruction-Gaussian-z50-e050.png' height = '200px'> </td>
	<td><img src = '/Implementations/Deep-Convolutional-GAN/results/Mean-reconstruction-Gaussian-z50-e075.png' height = '200px'> </td>
	<td><img src = '/Implementations/Deep-Convolutional-GAN/results/Mean-reconstruction-Gaussian-z50-e100.png' height = '200px'> </td>
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

