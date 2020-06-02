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
class MLP_relu(nn.Module):
    '''
        if you change MNIST model dimension, 
        You have to adjust dimensions setting in "training.py".
        
    '''
    def __init__(self):
        super(MLP_relu, self).__init__()

        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self,x):
        t1 = F.relu(self.fc1(x))
        t2 = F.relu(self.fc2(t1))
        t3 = self.fc3(t2).softmax(dim=1)
        return t1, t2, t3

class MLP_relu_parsing(nn.Module):
    
    def __init__(self):
        super(MLP_relu_parsing, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        t1 = self.fc1(x)
        t2 = F.relu(t1)
        t3 = self.fc2(t2)
        t4 = F.relu(t3)
        t5 = self.fc3(t4)
        t6 = F.softmax(t5)
        return t1, t2, t3, t4, t5, t6


class MLP_tanh(nn.Module):
    '''
        if you change MNIST model dimension, 
        You have to adjust dimensions setting in "training.py".
        
    '''
    def __init__(self):
        super(MLP_tanh, self).__init__()
        # self.fc1 = nn.Linear(784, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 10)

        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self,x):
        t1 = F.tanh(self.fc1(x))
        t2 = F.tanh(self.fc2(t1))
        t3 = F.softmax(self.fc3(t2)) # 下次放tanh看看
        return t1, t2, t3

class MLP_tanh_parsing(nn.Module):
    '''
        if you change MNIST model dimension, 
        You have to adjust dimensions setting in "training.py".
        
    '''
    def __init__(self):
        super(MLP_tanh_parsing, self).__init__()
        # self.fc1 = nn.Linear(784, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 10)

        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self,x):
        t1 = self.fc1(x)
        t2 = F.tanh(t1)
        t3 = self.fc2(t2)
        t4 = F.tanh(t3)
        t5 = self.fc3(t4)
        t6 = F.softmax(t5)

        return t1, t2, t3, t4, t5, t6
class MLP_tanh_parsing2(nn.Module):
    '''
        if you change MNIST model dimension, 
        You have to adjust dimensions setting in "training.py".
        
    '''
    def __init__(self):
        super(MLP_tanh_parsing2, self).__init__()
        # self.fc1 = nn.Linear(784, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 10)

        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self,x):
        t1 = self.fc1(x)
        t2 = F.tanh(t1)
        t3 = self.fc2(t2)
        t4 = F.tanh(t3)
        t5 = self.fc3(t4)
        t6 = F.softmax(t5)

        return t1, t2, t3, t4, t5, t6
        
class MLP_sigmoid(nn.Module):
    '''
        if you change MNIST model dimension, 
        You have to adjust dimensions setting in "training.py".
        
    '''
    def __init__(self):
        super(MLP_sigmoid, self).__init__()

        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self,x):
        t1 = F.sigmoid(self.fc1(x))
        t2 = F.sigmoid(self.fc2(t1))
        t3 = F.softmax(self.fc3(t2))
        return t1, t2, t3

class MLP_sigmoid_parsing(nn.Module):
    '''
        if you change MNIST model dimension, 
        You have to adjust dimensions setting in "training.py".
        
    '''
    def __init__(self):
        super(MLP_sigmoid_parsing, self).__init__()

        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self,x):
        t1 = self.fc1(x)
        t2 = F.sigmoid(t1)
        t3 = self.fc2(t2)
        t4 = F.sigmoid(t3)
        t5 = self.fc3(t4)
        t6 = F.softmax(t5)
        return t1, t2, t3, t4, t5, t6

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(10*10*20, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        t1 = F.dropout(x, p=0.5, training=self.training)
        # print(f"t1: {t1.shape}")
        x = F.relu(F.max_pool2d(self.conv2(t1), 2))
        t2 = F.dropout(x, p=0.5, training=self.training)
        # print(f"t2: {t2.shape}")
        
        x = t2.view(-1,10*10*20 )
        t3 = F.relu(self.fc1(x))
        # print(f"t3: {t3.shape}")

        x = F.dropout(t3, training=self.training)
        t4 = self.fc2(x)
        # print(f"t4: {t4.shape}")

        return t1, t2, t3, t4

class AA_MINEnet(nn.Module):
    def __init__(self, input_dim = 20):
        super(AA_MINEnet, self).__init__()

        self.fc1 = nn.Linear(input_dim, 1500)
        self.fc2 = nn.Linear(1500, 1500)
        self.fc3 = nn.Linear(1500, 1)

    def forward(self, x, t):
        # combine these two distributions
        t1 = time.time()
        input_mine = torch.cat((x, t),1)
        print(f"Concate elapsed time : {time.time() - t1}")
        h1 = F.leaky_relu(self.fc1(input_mine), negative_slope = 0.001)
        h2 = F.leaky_relu(self.fc2(h1), negative_slope = 0.001)
        h3 = self.fc3(h2)
        return h3

class MINEnet(nn.Module):
    def __init__(self, input_dim = 20):
        super(MINEnet, self).__init__()

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
