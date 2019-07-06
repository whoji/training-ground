# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from model import Generator, Discriminator
from data import dataloader


dataroot = "data/celeba"    # Root directory for dataset
workers = 2                 # Number of workers for dataloader
batch_size = 128            # Batch size during training
image_size = 64             # Spatial size of training images.
                    # All images will be resized to this size using a transformer.
nc = 3                      # Number of channels in the training images.
nz = 100                    # Size of z latent vector (i.e. size of generator input)
ngf = 64                    # Size of feature maps in generator
ndf = 64                    # Size of feature maps in discriminator
num_epochs = 5              # Number of training epochs
lr = 0.0002                 # LR
beta1 = 0.5                 # beta1 for Adam optimizers
ngpu = 1                    # Number of GPUs available. Use 0 for CPU mode.



###################
## testing


# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)



#################
## create net / model

# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)


# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)


###################
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))