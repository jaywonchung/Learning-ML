import torch
import torch.nn as nn

from constants import *


class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.ngpu = NGPU if torch.cuda.is_available() else 0
        self.dataset = DATASET

        modules = [
            nn.ConvTranspose2d(LATENT_DIM, NUM_FILTERS*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NUM_FILTERS*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(NUM_FILTERS*4, NUM_FILTERS*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NUM_FILTERS*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(NUM_FILTERS*2, NUM_FILTERS, 4, 2, 2 if DATASET == 'MNIST' else 1, bias=False),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.ReLU(inplace=True),
        ]
        if DATASET == 'CelebA':
            modules.extend([
                nn.ConvTranspose2d(NUM_FILTERS, NUM_FILTERS//2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(NUM_FILTERS//2),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(NUM_FILTERS//2, NUM_CHANNELS, 4, 2, 1, bias=False),
                nn.Tanh()
            ])
        else:
            modules.extend([
                nn.ConvTranspose2d(NUM_FILTERS, NUM_CHANNELS, 4, 2, 1, bias=False),
                nn.Tanh()
            ])
        self.layers = nn.Sequential(*modules)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
    
    def forward(self, z):
        return self.layers(z)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.ngpu = NGPU
        self.dataset = DATASET

        if DATASET == 'CelebA':
            modules = [
                nn.Conv2d(NUM_CHANNELS, NUM_FILTERS//2, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(NUM_FILTERS//2, NUM_FILTERS, 4, 2, 1, bias=False),
                nn.BatchNorm2d(NUM_FILTERS),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        else:
            modules = [
                nn.Conv2d(NUM_CHANNELS, NUM_FILTERS, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        modules.extend([
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS*2, 4, 2, 2 if DATASET == 'MNIST' else 1, bias=False),
            nn.BatchNorm2d(NUM_FILTERS*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(NUM_FILTERS*2, NUM_FILTERS*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NUM_FILTERS*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(NUM_FILTERS*4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        ])
        self.layers = nn.Sequential(*modules)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        return self.layers(x)