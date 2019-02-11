
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

## MNIST Generation



## CIFAR10 Reconstruction



- You can see that the sampled images are not of high quality. This was even worse for CIFAR10, so I did not generate sampled images for the CIFAR10 dataset.

## CIFAR10 Generation

# Usage
## Prerequisites

1. Pytorch and torchvision
2. Packages: numpy, matplotlib

## Execution

Command line:
```bash
python train.py -n True -b True -e 10 -ls 'CE' -lr 3e-4 -z 10 -p 1
```

Jupyter notebook:
```python
from train import main
%matplotlib inline

main(add_noise=True, binarize_input=True, epochs=10, loss='CE', learning_rate=3e-4, latent_dim=10, print_every=1)
```

## Arguments
Every argument is optional, and has a default value defined at ```arguments.py```.

- ```--add_noise, -n```: Whether to add dropout noise to input images. *Default*: - ```True```  
- ```--binarize_input, -b```: Whether to binarize input images (threshold 0.5). *Default*: ```True```
- ```--epochs, -e```: Number of epochs to train. *Default*: ```10```
- ```--loss, -ls```: Which loss function to use. Should be either ```'CE'``` or ```'MSE'```. *Default*: ```'CE'```
- ```--learning_rate, -lr```: Learning rate. This value is decayed to ```lr/10``` at epoch 6. *Default*: ```3e-4```
- ```--latent_dim, -z```: Dimension of the latent variable. *Default*: ```10```
- ```--print_every, -p```: How often to print training progress. *Default*: ```1```

Binarizing the input means that you model the output as a multinoulli distribution. Then using the cross-entropy loss is desirable in the Maximum Likelihood Estimation perspective. On the other hand if you do not binarize the input images, you would be modelling the output as a Multivariate Gaussian distribution. Then using the mean square error is desirable.
