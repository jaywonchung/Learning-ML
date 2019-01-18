# Resources
- [arXiv](https://arxiv.org/abs/1603.05279)
- [Official XNOR implementation of AlexNet](http://allenai.org/plato/xnornet)
- [My blog post(math formula displayed correctly)](https://jaywonchung.github.io/read/papers/XNOR-Nets)

# Abstract/Introduction
The two models presented:
> In Binary-Weight-Networks, the (convolution) filters are approximated with binary values resulting in 32 x memory saving.

> In XNOR-Networks, both the filters and the input to convolutional layers are binary. ... This results in 58 x faster convolutional operations...

Implications:
> XNOR-Nets offer the possibility of running state-of-the-art networks on CPUs (rather than GPUs) in real-time.

# Binary Convolutional Neural Networks
For future discussions we use the following mathematical notation for a CNN layer:  

$$ \mathcal{I}_{l(l=1,...,L)} = \mathbf{I}\in \mathbb{R} ^{c \times w_{\text{in}} \times h_{\text{in}}}$$  
$$ \mathcal{W}_{lk(k=1,...,K^l)}=\mathbf{W} \in \mathbb{R} ^{c \times w \times h} $$  
$$ \ast\text{ : convolution} $$  
$$ \oplus\text{ : convolution without multiplication} $$  
$$ \otimes \text{ : convolution with XNOR and bitcount} $$  
$$ \odot \text{ : elementwise multiplication} $$  

## Convolution with binary weights
In binary convolutional networks, we estimate the convolution filter weight as $$ \mathbf{W}\approx\alpha \mathbf{B} $$, where $$ \alpha $$ is a scalar scaling factor and $$ \mathbf{B} \in \{+1, -1\} ^{c \times w \times h} $$. Hence, we estimate the convolution operation as follows:  

$$ \mathbf{I} \ast \mathbf{W}\approx (\mathbf{I}\oplus \mathbf{B})\alpha $$  

To find an optimal estimation for $$ \mathbf{W}\approx\alpha \mathbf{B} $$ we solve the following problem:  

$$ J(\mathbf{B},\alpha)=\Vert \mathbf{W}-\alpha \mathbf{B}\Vert^2 $$

$$ \alpha ^*,\mathbf{B}^* =\underset{\alpha, \mathbf{B}}{\text{argmin}}J(\mathbf{B},\alpha) $$  

Going straight to the answer:  

$$ \alpha^* = \frac{1}{n}\Vert \mathbf{W}\Vert_{l1} $$  

$$ \mathbf{B}^*=\text{sign}(\mathbf{W}) $$  


## Training
The gradients are computed as follows:  

$$ \frac{\partial \text{sign}}{\partial r}=r \text{1}_{\vert r \vert \le1} $$  

$$ \frac{\partial L}{\partial \mathbf{W}_i}=\frac{\partial L}{\partial \widetilde{\mathbf{W}_i}}\left(\frac{1}{n} + \frac{\partial \text{sign}}{\partial \mathbf{W}_i}\alpha \right)$$

where $$\widetilde{\mathbf{W}}=\alpha \mathbf{B}$$, the estimated value of $$\mathbf{W}$$.

The gradient values are kepted as real values; they cannot be binarized due to excessive information loss. Optimization is done by either SGD with momentum or ADAM.

# XNOR-Networks
Convolutions are a set of dot products between a submatrix of the input and a filter. Thus we attempt to express dot products in terms of binary operations.
## Binary Dot Product
For vectors $$\mathbf{X}, \mathbf{W} \in \mathbb{R}^n$$ and $$\mathbf{H}, \mathbf{B} \in \{+1,-1\}^n$$, we approximate the dot product between $$\mathbf{X}$$ and $$\mathbf{W}$$ as

$$\mathbf{X}^\top \mathbf{W} \approx \beta \mathbf{H}^\top \alpha \mathbf{B}$$

We solve the following optimization problem:

$$\alpha^*, \mathbf{H}^*, \beta^*, \mathbf{B}^*=\underset{\alpha, \mathbf{H}, \beta, \mathbf{B}}{\text{argmin}} \Vert \mathbf{X} \odot \mathbf{W} - \beta \alpha \mathbf{H} \odot \mathbf{B} \Vert$$

Going straight to the answer:

$$\alpha^* \beta^*=\left(\frac{1}{n}\Vert \mathbf{X} \Vert_{l1}\right)\left(\frac{1}{n}\Vert \mathbf{W} \Vert_{l1}\right)$$

$$\mathbf{H}^* \odot \mathbf{B}^*=\text{sign}(\mathbf{X}) \odot \text{sign}(\mathbf{W})$$

## Convolution with binary inputs and weights
Calculating $$\alpha^* \beta^*$$ for every submatrix in input tensor $$\mathbf{I}$$ involves a large number of redundant computations. To overcome this inefficiency we first calculate

$$\mathbf{A}=\frac{\sum{\vert \mathbf{I}_{:,:,i} \vert}}{c}$$

which is an average over absolute values of $$\mathbf{I}$$ along its channel. Then, we convolve $$\mathbf{A}$$ with a 2D filter $$\mathbf{k} \in \mathbb{R}^{w \times h}$$ where $$\forall ij \ \mathbf{k}_{ij}=\frac{1}{w \times h}$$:

$$ \mathbf{K}=\mathbf{A} \ast \mathbf{k} $$

This $$\mathbf{K}$$ acts as a global $$\beta$$ spatially across the submatrices. Now we can estimate our convolution with binary inputs and weights as:

$$ \mathbf{I} \ast \mathbf{W} \approx (\text{sign}(\mathbf{I}) \otimes \text{sign}(\mathbf{W}) \odot \mathbf{K} \alpha$$

## Training
A CNN block in XNOR-Net has the following structure:

```
[Binary Normalization] - [Binary Activation] - [Binary Convolution] - [Pool]
```

The BinNorm layer normalizes the input batch by its mean and variance. The BinActiv layer calculates $$\mathbf{K}$$ and $$\text{sign}(\mathbf{I})$$. We may insert a non-linear activation function between the BinConv layer and the Pool layer.

# Experiments
The paper implemented the AlexNet, the Residual Net, and a GoogLenet variant(Darknet) with binary convolutions. This resulted in a few percent point of accuracy decrease, but overall worked fairly well. Refer to the paper for details.

# Discussion

Binary convolutions were not at all entirely binary; the gradients had to be real values. It would be fascinating if even the gradient is binarizable.
