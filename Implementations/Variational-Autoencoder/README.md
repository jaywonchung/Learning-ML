
# Variational Autoencoder for MNIST and CIFAR10

This is the pytorch implementation of:
- Variational Autoencoder (VAE)

which was introduced in [Auto-encoding Variational Bayes](https://arxiv.org/abs/1312.6114) by Kingma et al.

# Tutorial

Refer to my blog post:  
[Autoencoders, Denoising Autoencoders, and Variational Autoencoders](https://jaywonchung.github.io/study/machine-learning/Autoencoders/)

# Structure
## Model types

<table align='center'>
<tr align='center'>
	<td style="font-weight: bold"> Encoder </td>
	<td colspan=3>  Gaussian </td>
</tr>
<tr align='center'>
	<td style="font-weight: bold"> Decoder </td>
	<td colspan=2> Gaussian </td>
	<td> Bernoulli </td>
</tr>
<tr align='center'>
	<td style="font-weight: bold"> Model output </td>
	<td> Mean, Std</td>
	<td> Mean</td>
	<td> Probability </td>
</tr>
<tr align='center'>
	<td style="font-weight: bold"> Supported dataset </td>
	<td colspan=2> MNIST, CIFAR10</td>
	<td> MNIST</td>
</tr>
<tr align='center'>
	<td style="font-weight: bold"> Input image</td>
	<td colspan=2> As is </td>
	<td>  Binarized</td>
</tr>
</table>

## Network Structure

Refer to [```model.py```](/Implementations/Variational-Autoencoder/model.py). At ```__init__```, I annotated the shape change of the input every layer.

# Results
## MNIST Reconstruction

Training specs:
- Initial learning rate ```1e-4``` with ```torch.optim.Adam``` optimizer
- Learning rate scheduled with ```torch.optim.lr_scheduler.ReduceLROnPlateau``` by mean training loss
- Batch size ```64```

With **Bernoulli decoder** and **2-D** latent variable:

<table align='center'>
<tr align='center'>
    <td> Input image </td>
</tr>
<tr align='center'>
    <td><img src = '/Implementations/Variational-Autoencoder/results/Binarized-truth-Bernoulli-z10.png' height = '200px'> </td>
</tr>
<tr align='center'>
	<td> Epoch 1 </td>
    <td> Epoch 10  </td>
    <td> Epoch 30 </td>
    <td> Epoch 50 </td>
</tr>
<tr align='center'>
	<td><img src = '/Implementations/Variational-Autoencoder/results/Mean-reconstruction-Bernoulli-z2-e001.png' height = '200px'>
    <td><img src = '/Implementations/Variational-Autoencoder/results/Mean-reconstruction-Bernoulli-z2-e010.png' height = '200px'>
    <td><img src = '/Implementations/Variational-Autoencoder/results/Mean-reconstruction-Bernoulli-z2-e030.png' height = '200px'>
    <td><img src = '/Implementations/Variational-Autoencoder/results/Mean-reconstruction-Bernoulli-z2-e050.png' height = '200px'>
</tr>
<tr align='center'>
	<td><img src = '/Implementations/Variational-Autoencoder/results/Sampled-reconstruction-Bernoulli-z2-e001.png' height = '200px'>
    <td><img src = '/Implementations/Variational-Autoencoder/results/Sampled-reconstruction-Bernoulli-z2-e010.png' height = '200px'>
    <td><img src = '/Implementations/Variational-Autoencoder/results/Sampled-reconstruction-Bernoulli-z2-e030.png' height = '200px'>
    <td><img src = '/Implementations/Variational-Autoencoder/results/Sampled-reconstruction-Bernoulli-z2-e050.png' height = '200px'>
</tr>
</table>

With **Bernoulli decoder** and **10-D** latent variable:

<table align='center'>
<tr align='center'>
    <td> Input image </td>
</tr>
<tr align='center'>
    <td><img src = '/Implementations/Variational-Autoencoder/results/Binarized-truth-Bernoulli-z2.png' height = '200px'> </td>
</tr>
<tr align='center'>
	<td> Epoch 1 </td>
    <td> Epoch 10  </td>
    <td> Epoch 20 </td>
    <td> Epoch 30 </td>
</tr>
<tr align='center'>
	<td><img src = '/Implementations/Variational-Autoencoder/results/Mean-reconstruction-Bernoulli-z10-e001.png' height = '200px'>
    <td><img src = '/Implementations/Variational-Autoencoder/results/Mean-reconstruction-Bernoulli-z10-e010.png' height = '200px'>
    <td><img src = '/Implementations/Variational-Autoencoder/results/Mean-reconstruction-Bernoulli-z10-e020.png' height = '200px'>
    <td><img src = '/Implementations/Variational-Autoencoder/results/Mean-reconstruction-Bernoulli-z10-e030.png' height = '200px'>
</tr>
<tr align='center'>
	<td><img src = '/Implementations/Variational-Autoencoder/results/Sampled-reconstruction-Bernoulli-z10-e001.png' height = '200px'>
    <td><img src = '/Implementations/Variational-Autoencoder/results/Sampled-reconstruction-Bernoulli-z10-e010.png' height = '200px'>
    <td><img src = '/Implementations/Variational-Autoencoder/results/Sampled-reconstruction-Bernoulli-z10-e020.png' height = '200px'>
    <td><img src = '/Implementations/Variational-Autoencoder/results/Sampled-reconstruction-Bernoulli-z10-e030.png' height = '200px'>
</tr>
</table>

- Row 2 are images generated directly from model output ```p```, while row 3 are sampled pixel-by-pixel from a Bernoulli distribution parametrized by ```p```.   
- You can see that the sampled images are not of high quality. This was even worse for CIFAR10, so I did not generate sampled images for the CIFAR10 dataset.

## MNIST Generation

With **2-D uniformly sampled** latent variables:

<img src = '/Implementations/Variational-Autoencoder/results/uniform-generation-MNIST-400.png' height = '450px'>

With **2-D randomly sampled** latent variables:

<img src = '/Implementations/Variational-Autoencoder/results/random-generation-MNIST-400.png' height = '450px'>

## CIFAR10 Reconstruction

With **Gaussian decoder** and **30-D** latent variable:

<table align='center'>
<tr align='center'>
    <td> Input image </td>
    <td> Epoch 1 </td>
    <td> Epoch 10 </td>
    <td> Epoch 50 </td>
</tr>
<tr align='center'>
    <td><img src = '/Implementations/Variational-Autoencoder/results/Truth-Gaussian-z30.png' height = '200px'> </td>
    <td><img src = '/Implementations/Variational-Autoencoder/results/Mean-reconstruction-Gaussian-z30-e001.png' height = '200px'> </td>
    <td><img src = '/Implementations/Variational-Autoencoder/results/Mean-reconstruction-Gaussian-z30-e010.png' height = '200px'> </td>
    <td><img src = '/Implementations/Variational-Autoencoder/results/Mean-reconstruction-Gaussian-z30-e050.png' height = '200px'> </td>
</tr>
<tr align='center'>
	<td> Epoch 100 </td>
    <td> Epoch 300  </td>
    <td> Epoch 600 </td>
    <td> Epoch 1000 </td>
</tr>
<tr align='center'>
	<td><img src = '/Implementations/Variational-Autoencoder/results/Mean-reconstruction-Gaussian-z30-e100.png' height = '200px'> </td>
	<td><img src = '/Implementations/Variational-Autoencoder/results/Mean-reconstruction-Gaussian-z30-e300.png' height = '200px'> </td>
	<td><img src = '/Implementations/Variational-Autoencoder/results/Mean-reconstruction-Gaussian-z30-e600.png' height = '200px'> </td>
	<td><img src = '/Implementations/Variational-Autoencoder/results/Mean-reconstruction-Gaussian-z30-e1000.png' height = '200px'> </td>
</tr>
</table>

## CIFAR10 Generation

With **30-D** latent variables **uniformly** varied on two specific dimensions:

<table align='center'>
<tr align='center'>
	<td><img src = '/Implementations/Variational-Autoencoder/results/uniform-generation-CIFAR10-400-1.png' height = '300px'> </td>
	<td><img src = '/Implementations/Variational-Autoencoder/results/uniform-generation-CIFAR10-400-2.png' height = '300px'> </td>
	<td><img src = '/Implementations/Variational-Autoencoder/results/uniform-generation-CIFAR10-400-3.png' height = '300px'> </td>
</tr>
</table>

With **30-D** latent variables **randomly** varied on two specific dimensions:

<img src = '/Implementations/Variational-Autoencoder/results/random-generation-CIFAR10-400.png' height = '300px'>

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

