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
        t3 = self.fc3(t2) # 下次放tanh看看
        return t1, t2, t3


class AA_MINEnet(nn.Module):
    def __init__(self, input_dim = 20):
        super(AA_MINEnet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x, t):
        # combine these two distributions
        input_mine = torch.cat((x, t),1)
        # print(f"input_mine.shape = {input_mine.shape}")
        h1 = F.relu(self.fc1(input_mine))
        # print(f"h1.shape={h1.shape}")
        h2 = F.relu(self.fc2(h1))
        # print(f"h2.shape={h2.shape}")
        h3 = F.relu(self.fc3(h2))
        # print(f"h3.shape={h3.shape}")
        h4 = self.fc4(h3)        
        return h4

class MINEnet(nn.Module):
    def __init__(self, input_dim = 20):
        super(MINEnet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x, t):
        # combine these two distributions
        input_mine = torch.cat((x, t),1)
        h1 = F.relu(self.fc1(input_mine))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = self.fc4(h3)        
        return h4


if __name__ == "__main__":
    net = Net().to(device)
    print(net)