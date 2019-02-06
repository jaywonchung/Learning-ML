# Denoising Autoencoder for MNIST

This is the pytorch implementation of:
- Autoencoder (AE)
- Denoising Autoencoder (DAE)

# Tutorial

Refer to my blog post:  
[Autoencoders, Denoising Autoencoders, and Variational Autoencoders](https://jaywonchung.github.io/study/machine-learning/Autoencoders/)

# Results
## Autoencoder

Training specs:
- Binarized input images
- Cross-entropy loss
- 10-D latent variable

<table align='center'>
<tr align='center'>
    <td> Input image </td>
    <td> Epoch 1 </td>
    <td> Epoch 5 </td>
    <td> Epoch 10 </td>
</tr>
<tr>
    <td><img src = 'results/Binarized Truth.png' height = '150px'>
    <td><img src = 'results/AE-CE-z10-e1.png' height = '150px'>
    <td><img src = 'results/AE-CE-z10-e5.png' height = '150px'>
    <td><img src = 'results/AE-CE-z10-e10.png' height = '150px'>
</tr>
</table>

## Denoising Autoencoder

Training specs:
- Binarized input images
- Dropout noise (p=0.5) applied to input images

With 2-D latent variable:

<table align='center'>
<tr align='center'>
    <td> Input image </td>
    <td> Noised input image </td>
</tr>
<tr align='center'>
    <td><img src = 'results/Binarized Truth.png' height = '200px'>
    <td><img src = 'results/Noised Truth.png' height = '200px'>
</tr>
<tr align='center'>
    <td> Epoch 1 </td>
    <td> Epoch 10 </td>
    <td> Epoch 20 </td>
    <td> Epoch 30 </td>
</tr>
<tr align='center'>
    <td><img src = 'results/DAE-CE-z2-e1-nobin.png' height = '200px'>
    <td><img src = 'results/DAE-CE-z2-e10-nobin.png' height = '200px'>
    <td><img src = 'results/DAE-CE-z2-e20-nobin.png' height = '200px'>
    <td><img src = 'results/DAE-CE-z2-e30-nobin.png' height = '200px'>
</tr>
</table>

With 10-D latent variable:

<table align='center'>
<tr align='center'>
    <td> Input image </td>
    <td> Noised input image </td>
</tr>
<tr align='center'>
    <td><img src = 'results/Binarized Truth-nobin.png' height = '200px'>
    <td><img src = 'results/Noised Truth-nobin.png' height = '200px'>
</tr>
<tr align='center'>
    <td> Epoch 1 </td>
    <td> Epoch 2 </td>
    <td> Epoch 5 </td>
    <td> Epoch 10 </td>
</tr>
<tr align='center'>
    <td><img src = 'results/DAE-CE-z10-e1-nobin.png' height = '200px'>
    <td><img src = 'results/DAE-CE-z10-e2-nobin.png' height = '200px'>
    <td><img src = 'results/DAE-CE-z10-e5-nobin.png' height = '200px'>
    <td><img src = 'results/DAE-CE-z10-e10-nobin.png' height = '200px'>
</tr>
</table>