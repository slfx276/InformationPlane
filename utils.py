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
import argparse
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

def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser('Create inputs of main.py')


    parser.add_argument("-bs", "--batchsize", type=int, default = 1024, dest = "batch_size", 
                            help="set training batch size of MNIST model")

    parser.add_argument("-e", "--mnistepoch", type=int, default=10, dest = "mnist_epoch",
                            help="set training epochs of MNIST model")

    parser.add_argument("-var", "--noisevariance", type=float, default=0.2, dest = "noise_var",
                            help="noise variance in noisy representation while using MINE ")

    parser.add_argument("-mie", "--mineepoch", type=int, default=130, dest = "mine_epoch",
                            help="training epochs of MINE model while estimating mutual information")

    parser.add_argument("-amie", "--amineepoch", type=int, default=150, dest = "aamine_epoch",
                            help="how many batch do you want to combined into a group in order to calculate MI")

    # 59 because 1024*59 > 60000
    parser.add_argument("-bg", "--bgroup", type=int, default=59, dest = "batch_group", 
                            help="how many batch do you want to combined into a group in order to calculate MI")

    parser.add_argument("-f", "--folder", type=str, default="mine", dest="folder_name", 
                            help="the name of folder which you create for saving MINE training trend.")

    parser.add_argument("-opt", "--optimizer", type=str, default="sgd", dest="mnist_opt", 
                            help="the optimizer used to train MNIST model.")
                
    parser.add_argument("-lr", "--lr", type=float, default = 0.001, dest = "mnist_lr", 
                            help="initial learning rate used to train MNIST model.")
    
    parser.add_argument("-re", "--retrain", action="store_true", dest="retrain", 
                            help="Retrain MNIST model and then store new representations")

    parser.add_argument("-show", "--showmine",  action="store_true", dest="show", 
                            help="show and save MINE training trend. (need GUI)")
                            
    parser.add_argument("-cls", "--cleanfile",  action="store_true", dest="clean_old_files", 
                            help="clean old data before creating new ones")

    return parser.parse_args()