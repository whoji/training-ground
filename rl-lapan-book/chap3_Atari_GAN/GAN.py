#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

import numpy as np
from tensorboardX import SummaryWriter

LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 16
LEARNING_RATE = 0.0001


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # this pipe converges image into the single number
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS, out_channels=DISCR_FILTERS*2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 2, out_channels=DISCR_FILTERS * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 4, out_channels=DISCR_FILTERS * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        return conv_out.view(-1, 1).squeeze(dim=1) # wtf is this doing ?


class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        # pipe deconvolves input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=GENER_FILTERS * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 8, out_channels=GENER_FILTERS * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 4, out_channels=GENER_FILTERS * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 2, out_channels=GENER_FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)


class Trainer():
    def __init__(self, input_shape, device):
        self.net_discr = Discriminator(input_shape=input_shape).to(device)
        self.net_gener = Generator(output_shape=input_shape).to(device)

        self.objective = nn.BCELoss()
        self.gen_optimizer = optim.Adam(params=self.net_gener.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        self.dis_optimizer = optim.Adam(params=self.net_discr.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        self.writer = SummaryWriter()

        self.gen_losses = []
        self.dis_losses = []
        self.iter_no = 0

        self.batch_v = None

    def gen_fake(self, device):
        gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1).normal_(0, 1).to(device)
        gen_output_v = self.net_gener(gen_input_v)
        return gen_output_v

    def train_discr(self, gen_output_v, true_labels_v, fake_labels_v):
        self.dis_optimizer.zero_grad()
        dis_output_true_v = self.net_discr(self.batch_v)
        dis_output_fake_v = self.net_discr(gen_output_v.detach())
        dis_loss = self.objective(dis_output_true_v, true_labels_v) + self.objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        self.dis_optimizer.step()
        self.dis_losses.append(dis_loss.item())

    def train_gener(self, gen_output_v, true_labels_v):
        self.gen_optimizer.zero_grad()
        dis_output_v = self.net_discr(gen_output_v)
        gen_loss_v = self.objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        self.gen_optimizer.step()
        self.gen_losses.append(gen_loss_v.item())

    def write_summary(self):
        self.writer.add_scalar("gen_loss", np.mean(self.gen_losses), self.iter_no)
        self.writer.add_scalar("dis_loss", np.mean(self.dis_losses), self.iter_no)
        self.gen_losses = []
        self.dis_losses = []

    def save_images(self, gen_output_v):
        self.writer.add_image("fake", vutils.make_grid(gen_output_v.data[:64]), self.iter_no)
        self.writer.add_image("real", vutils.make_grid(self.batch_v.data[:64]), self.iter_no)