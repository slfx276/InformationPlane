import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# import holoviews as hv
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import time
import logging
import sys

from model import MNIST_Net

def training_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device

def dataLoader(batch_size = 256):
    # Transform
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,)),])
    #Data
    trainSet = datasets.MNIST(root='MNIST', download=True, train=True, 
                            transform=transform)
    testSet = datasets.MNIST(root='MNIST', download=True, train=False,
                            transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size,
                                            shuffle=True)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=batch_size,
                                            shuffle=False)

    return trainLoader, testLoader

def Create_Logger(name = __name__, log_level = logging.DEBUG):

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create STDERR handler
    handler = logging.StreamHandler(sys.stderr)
    # ch.setLevel(logging.DEBUG)

    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Set STDERR handler as the only handler 
    logger.handlers = [handler]
    return logger