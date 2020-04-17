import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# import holoviews as hv
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import time

# define MNIST classification model
class MNIST_Net(nn.Module):
    '''
        if you change MNIST model dimension, 
        You have to adjust dimensions setting in "training.py".
        
    '''
    def __init__(self):
        super(MNIST_Net, self).__init__()
        # self.fc1 = nn.Linear(784, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 10)

        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self,x):
        t1 = F.relu(self.fc1(x))
        t2 = F.relu(self.fc2(t1))
        t3 = self.fc3(t2) 
        return t1, t2, t3

        # t1 = F.sigmoid(self.fc1(x))
        # t2 = F.sigmoid(self.fc2(t1))
        # t3 = self.fc3(t2)
        # return t1, t2, t3

        # while using tanh, the learning rate sould be small, ex: lr = 0.001
        # t1 = F.tanh(self.fc1(x))
        # t2 = F.tanh(self.fc2(t1))
        # t3 = self.fc3(t2)
        # return t1, t2, t3
        


class AA_MINEnet(nn.Module):
    def __init__(self, input_dim = 20):
        super(AA_MINEnet, self).__init__()
        #self.fc1 = nn.Linear(input_dim, 128)
        #self.fc2 = nn.Linear(128, 64)
        #self.fc3 = nn.Linear(64, 64)
        #self.fc4 = nn.Linear(64, 1)

        self.fc1 = nn.Linear(input_dim, 1500)
        self.fc2 = nn.Linear(1500, 1500)
        self.fc3 = nn.Linear(1500, 1)

    def forward(self, x, t):
        # combine these two distributions
        input_mine = torch.cat((x, t),1)
        h1 = F.leaky_relu(self.fc1(input_mine), negative_slope = 0.001)
        h2 = F.leaky_relu(self.fc2(h1), negative_slope = 0.001)
        h3 = self.fc3(h2)
        return h3

class MINEnet(nn.Module):
    def __init__(self, input_dim = 20):
        super(MINEnet, self).__init__()
        #self.fc1 = nn.Linear(input_dim, 128)
        #self.fc2 = nn.Linear(128, 64)
        #self.fc3 = nn.Linear(64, 64)
        #self.fc4 = nn.Linear(64, 1)

        self.fc1 = nn.Linear(input_dim, 1500)
        self.fc2 = nn.Linear(1500, 1500)
        self.fc3 = nn.Linear(1500, 1)

    def forward(self, x, t):
        # combine these two distributions
        input_mine = torch.cat((x, t),1)
        h1 = F.leaky_relu(self.fc1(input_mine), negative_slope = 0.001)
        h2 = F.leaky_relu(self.fc2(h1), negative_slope = 0.001)
        h3 = self.fc3(h2)
        return h3

