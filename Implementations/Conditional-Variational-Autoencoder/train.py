import sys
import datetime

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

from model import CVAE
from plot_utils import *
from arguments import get_args, defaults

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(**kwargs):
    """
    Main function that trains the model
    1. Retrieve arguments from kwargs
    2. Prepare data
    3. Train
    4. Display/save first batch of training set (truth and reconstructed) after every epoch
    
    Args:
        dataset: Which dataset to use
        decoder_type: How to model the output pixels, Gaussian or Bernoulli
        model_sigma: In case of Gaussian decoder, whether to model the sigmas too
        epochs: How many epochs to train model
        batch_size: Size of training / testing batch
        lr: Learning rate
        latent_dim: Dimension of latent variable
        print_every: How often to print training progress
        resume_path: The path of saved model with which to resume training
        resume_epoch: In case of resuming, the number of epochs already done 

    Notes:
        - Saves model to folder 'saved_model/' every 20 epochs and when done
        - Capable of training from scratch and resuming (provide saved model location to argument resume_path)
        - Schedules learning rate with optim.lr_scheduler.ReduceLROnPlateau
            : Decays learning rate by 1/10 when mean loss of all training data does not decrease for 10 epochs
    """
    # Retrieve arguments
    dataset = kwargs.get('dataset', defaults['dataset'])
    decoder_type = kwargs.get('decoder_type', defaults['decoder_type'])
    if decoder_type == 'Gaussian':
        model_sigma = kwargs.get('model_sigma', defaults['model_sigma'])
    epochs = kwargs.get('epochs', defaults['epochs'])
    batch_size = kwargs.get('batch_size', defaults['batch_size'])
    lr = kwargs.get('learning_rate', defaults['learning_rate'])
    latent_dim = kwargs.get('latent_dim', defaults['latent_dim'])
    print_every = kwargs.get('print_every', defaults['print_every'])
    resume_path = kwargs.get('resume_path', defaults['resume_path'])
    resume_epoch = kwargs.get('resume_epoch', defaults['resume_epoch'])
    
    # Specify dataset transform on load
    if decoder_type == 'Bernoulli':
        trsf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x : (x >= 0.5).float())])
    elif decoder_type == 'Gaussian':
        trsf = transforms.ToTensor()

    # Load dataset with transform
    if dataset == 'MNIST':
        train_data = datasets.MNIST(
            root='MNIST', train=True, transform=trsf, download=True)
        test_data = datasets.MNIST(
            root='MNIST', train=False, transform=trsf, download=True)
    elif dataset == 'CIFAR10':
        train_data = datasets.CIFAR10(
            root='CIFAR10', train=True, transform=trsf, download=True)
        test_data = datasets.CIFAR10(
            root='CIFAR10', train=False, transform=trsf, download=True)
    
    # Instantiate dataloader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # Instantiate/Load model and optimizer
    if resume_path:
        autoencoder = torch.load(resume_path, map_location=device)
        optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
        print('Loaded saved model at ' + resume_path)
    else:
        if decoder_type == 'Bernoulli':
            autoencoder = CVAE(latent_dim, dataset, decoder_type).to(device)
        else:
            autoencoder = CVAE(latent_dim, dataset, decoder_type, model_sigma).to(device)
        optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    
    # Instantiate learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    
    # Announce current mode
    print(f'Start training CVAE with Gaussian encoder and {decoder_type} decoder on {dataset} dataset from epoch {resume_epoch+1}')

    # Prepare batch to display with plt
    first_test_batch, first_test_batch_label = iter(test_loader).next()
    first_test_batch, first_test_batch_label = first_test_batch.to(device), first_test_batch_label.to(device)

    # Train
    autoencoder.train()
    for epoch in range(resume_epoch, epochs+resume_epoch):
        loss_hist = []
        for batch_ind, (input_data, input_label) in enumerate(train_loader):
            input_data, input_label = input_data.to(device), input_label.to(device)
            
            # Forward propagation
            if decoder_type == 'Bernoulli':
                z_mu, z_sigma, p = autoencoder(input_data, input_label)
            elif model_sigma:
                z_mu, z_sigma, out_mu, out_sigma = autoencoder(input_data, input_label)
            else:
                z_mu, z_sigma, out_mu = autoencoder(input_data, input_label)

            # Calculate loss
            KL_divergence_i = 0.5 * torch.sum(z_mu**2 + z_sigma**2 - torch.log(1e-8+z_sigma**2) - 1., dim=1)
            if decoder_type == 'Bernoulli':
                reconstruction_loss_i = -torch.sum(F.binary_cross_entropy(p, input_data, reduction='none'), dim=(1,2,3))
            elif model_sigma:
                reconstruction_loss_i = -0.5 * torch.sum(torch.log(1e-8+6.28*out_sigma**2) + ((input_data-out_mu)**2)/(out_sigma**2), dim=(1,2,3))
            else:
                reconstruction_loss_i = -0.5 * torch.sum((input_data-out_mu)**2, dim=(1,2,3))
            ELBO_i = reconstruction_loss_i - KL_divergence_i
            loss = -torch.mean(ELBO_i)

            loss_hist.append(loss)
            
            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            
            # Update parameters
            optimizer.step()

            # Print progress
            if batch_ind % print_every == 0:
                train_log = 'Epoch {:03d}/{:03d}\tLoss: {:.6f}\t\tTrain: [{}/{} ({:.0f}%)]           '.format(
                    epoch+1, epochs+resume_epoch, loss.cpu().item(), batch_ind+1, len(train_loader),
                                100. * batch_ind / len(train_loader))
                print(train_log, end='\r')
                sys.stdout.flush()
        

        # Learning rate decay
        scheduler.step(sum(loss_hist)/len(loss_hist))

        # Save model every 20 epochs
        if (epoch+1)%20 == 0 and epoch+1!=epochs:
            PATH = f'saved_model/{dataset}-{decoder_type}-e{epoch+1}-z{latent_dim}' + datetime.datetime.now().strftime("-%b-%d-%H-%M-%p")
            torch.save(autoencoder, PATH)
            print('\vTemporarily saved model to ' + PATH)

        # Display training result with test set
        data = f'-{decoder_type}-z{latent_dim}-e{epoch+1:03d}'
        with torch.no_grad():

            display_and_save_latent(autoencoder.z, '-{epoch}')

            if decoder_type == 'Bernoulli':
                z_mu, z_sigma, p = autoencoder(first_test_batch, first_test_batch_label)
                output = torch.bernoulli(p)

                display_and_save_batch("Binarized-truth", first_test_batch, data, save=(epoch==0))
                display_and_save_batch("Mean-reconstruction", p, data, save=True)
                display_and_save_batch("Sampled-reconstruction", output, data, save=True)

            elif model_sigma:
                z_mu, z_sigma, out_mu, out_sigma = autoencoder(first_test_batch, first_test_batch_label)
                output = torch.normal(out_mu, out_sigma).clamp(0., 1.)

                display_and_save_batch("Truth", first_test_batch, data, save=(epoch==0))
                display_and_save_batch("Mean-reconstruction", out_mu, data, save=True)
                # display_and_save_batch("Sampled reconstruction", output, data, save=True)

            else:
                z_mu, z_sigma, out_mu = autoencoder(first_test_batch, first_test_batch_label)
                output = torch.normal(out_mu, torch.ones_like(out_mu)).clamp(0., 1.)

                display_and_save_batch("Truth", first_test_batch, data, save=(epoch==0))
                display_and_save_batch("Mean-reconstruction", out_mu, data, save=True)
                # display_and_save_batch("Sampled reconstruction", output, data, save=True)

    # Save final model
    PATH = f'saved_model/{dataset}-{decoder_type}-e{epochs+resume_epoch}-z{latent_dim}' + datetime.datetime.now().strftime("-%b-%d-%H-%M-%p")
    torch.save(autoencoder, PATH)
    print('\vSaved model to ' + PATH)


if __name__ == "__main__":
    args = get_args()
    main(**vars(args))