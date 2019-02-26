import torch
import torch.nn as nn

from constants import *


class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.ngpu = NGPU
        self.layers = nn.Sequential(
            # [N x 100 x 1 x 1]
            nn.ConvTranspose2d(LATENT_DIM, NUM_FILTERS*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NUM_FILTERS*8),
            nn.ReLU(inplace=True),
            # [N x 512 x 4 x 4]
            nn.ConvTranspose2d(NUM_FILTERS*8, NUM_FILTERS*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NUM_FILTERS*4),
            nn.ReLU(inplace=True),
            # [N x 256 x 8 x 8]
            nn.ConvTranspose2d(NUM_FILTERS*4, NUM_FILTERS*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NUM_FILTERS*2),
            nn.ReLU(inplace=True),
            # [N x 128 x 16 x 16]
            nn.ConvTranspose2d(NUM_FILTERS*2, NUM_FILTERS, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.ReLU(inplace=True),
            # [N x 64 x 32 x 32]
            nn.ConvTranspose2d(NUM_FILTERS, NUM_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
            # [N x 3 x 64 x 64]
        )

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
        self.layers = nn.Sequential(
            # [N x 3 x 64 x 64]
            nn.Conv2d(NUM_CHANNELS, NUM_FILTERS, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # [N x 64 x 32 x 32]
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NUM_FILTERS*2),
            nn.LeakyReLU(0.2, inplace=True),
            # [N x 128 x 16 x 16]
            nn.Conv2d(NUM_FILTERS*2, NUM_FILTERS*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NUM_FILTERS*4),
            nn.LeakyReLU(0.2, inplace=True),
            # [N x 256 x 8 x 8]
            nn.Conv2d(NUM_FILTERS*4, NUM_FILTERS*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NUM_FILTERS*8),
            nn.LeakyReLU(0.2, inplace=True),
            # [N x 512 x 4 x 4]
            nn.Conv2d(NUM_FILTERS*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # [N x 1 x 1 x 1]
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        return self.layers(x)