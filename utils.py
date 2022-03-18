# ---- torch imports --- #
import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

# --- general system imports --- #
from typing import Dict, Any

import pandas as pd
import numpy as np

import os
import sys
import pickle
import struct
from array import array

# --- image processing imports --- #
import png
from PIL import Image
from PIL import ImageOps

import matplotlib.pyplot as plt
import numpy as np


# --- utility functions --- #

# visualize images as grid
def imshow(inp, title=None, normalize=False, figsize=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    
    if figsize is None:
        figsize = (20, 10)

    if normalize == True:
        # normalization may be required in some cases, but not here
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
    
    plt.figure(figsize = figsize)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X_cpu, y_cpu) in enumerate(dataloader):
        # Compute prediction and loss
        X = X_cpu.to(device) # must put model and data both on gpu (if available)
        y = y_cpu.to(device)
        pred = model(X)
        pred = torch.squeeze(pred)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        # evaluate loss by looping once over entire testing set
        # ensure no gradients are computed
        for X_cpu, y_cpu in dataloader:
            X = X_cpu.to(device) # must put model and data both on gpu (if available)
            y = y_cpu.to(device)

            pred = model(X)
            pred = torch.squeeze(pred)
            test_loss += loss_fn(pred, y).item()
    
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


def train(model, loss_fn, optimizer, trainl, testl, epochs=None):
    """ Trains model by minimizing loss_fn using optimizer.
        Trains on trainl data and tests occassionally on testl data.
        Loops over training dataset for # determined by epochs.
        No returns; the model is trained in-place.
    """
    # training happens here; can take a long time
    if epochs is None:
        epochs = 50
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(trainloader, model, loss_fn, optimizer)
        test_loop(testloader, model, loss_fn)
    print("Done!")