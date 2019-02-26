import sys

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision.utils as utils
import torchvision.transforms as transforms

from constants import *
from model import *
from dataset import create_dataloader
from plot_utils import display_batch

device = torch.device('cuda:0' if torch.cuda.is_available() and NGPU > 0 else 'cpu')


def main(**kwargs):

    # Instantiate DataLoader
    dataloader = create_dataloader(name='celeba')

    # Instantiate Model
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Multi-gpu is desired
    if device.type == 'cuda' and NGPU > 0:
        generator = nn.DataParallel(generator, list(range(NGPU)))
        discriminator = nn.DataParallel(discriminator, list(range(NGPU)))
    
    # Set criterion function
    criterion = F.binary_cross_entropy

    # Set reference latent vector for training-time evaluation
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)

    # Instantiate optimizer
    optim_generator = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))

    # Training status tracker
    img_list = []

    # Announce training
    print("Begin training DCGAN")

    # Train
    for epoch in range(EPOCHS):
        for i, real_data in enumerate(dataloader, 0):

            # Format real image batch
            real_data = real_data[0].to(device)
            label = torch.full((BATCH_SIZE,), REAL_LABEL, device=device)
            
            # Fowrard pass discriminator
            disc_output = discriminator(real_data).squeeze()

            # Calculate discriminator real_loss
            disc_real_loss = criterion(disc_output, label)

            # Backward propagate real_loss
            discriminator.zero_grad()
            disc_real_loss.backward()

            # Generate fake image batch
            noise = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1, device=device)
            fake_data = generator(noise)
            label.fill_(FAKE_LABEL)

            # Forward pass generator
            disc_output = discriminator(fake_data.detach()).squeeze()

            # Calculate discriminator fake_loss and loss
            disc_fake_loss = criterion(disc_output, label)
            disc_loss = disc_real_loss + disc_fake_loss

            # Backward propagate fake_loss
            disc_fake_loss.backward()

            # Update discriminator parameters
            optim_discriminator.step()

            # Forward pass fake_data to updated discriminator
            disc_output = discriminator(fake_data).squeeze()

            # Fill label in the generator's perspective
            label.fill_(REAL_LABEL)

            # Calculate generator loss
            gen_loss = criterion(disc_output, label)

            # Backward propagate generator
            generator.zero_grad()
            gen_loss.backward()

            # Update generator parameters
            optim_generator.step()

            # Print training status
            if i % PRINT_EVERY == 0:
                message = f'Epochs: {epoch+1:02d}/{EPOCHS:02d}\tBatch: {i+1:03d}/{len(dataloader):03d}\tdisc_loss: {disc_loss.item():.4f}\tgen_loss: {gen_loss.item():.4f}'
                print(message, end='\r')
                sys.stdout.flush()
        
        # Evaluate training status by generating images on a fixed noise
        with torch.no_grad():
            fake_image = generator(fixed_noise).dtach().cpu()
        img_list.append(utils.make_grid(fake_image, padding=2, normalize=True))

        # Create checkpoints
        torch.save(discriminator, f'saved_model/discriminator_e{epoch:02d}')
        torch.save(generator, f'saved_model/generator_e{epoch:02d}')


if __name__ == "__main__":
    main()