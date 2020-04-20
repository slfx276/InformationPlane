import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# import holoviews as hv
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import time
import os
import pickle

import model
import utils



logger = utils.Create_Logger(__name__)

def mnist_training(batch_size, mnist_epochs, Retrain = False, lr = 0.001,
                                         opt = "adam", model_type = "mlprelu"):

    time1 = time.time()

    device = utils.training_device()
    conv_idx = list()
    if model_type == "mlptanh":
        mnist_net = model.MLP_tanh().to(device)
        dimensions = [500, 256, 10] # You have to adjust this if you change MNIST model dimension
    elif model_type == "mlprelu":
        mnist_net = model.MLP_relu().to(device)
        dimensions = [500, 256, 10] # You have to adjust this if you change MNIST model dimension
    elif model_type == "cnn":
        mnist_net = model.CNN().to(device)
        conv_idx = [0, 1] # which layer is convolutional layer
        dimensions = [10*24*24, 20*10*10, 256, 10]
        
    # create storage container of hidden layer representations 
    num_layers = len(dimensions)

    label_y = [np.empty(shape=[0, 10]) for i in range(mnist_epochs)]

    all_repre = []
    for layer_idx in range(num_layers):
        all_repre.append([])
        for epoch in range(mnist_epochs):
            all_repre[layer_idx].append(np.empty(shape = [0, dimensions[layer_idx]]))

    # save training model config
    with open("mnist_net_config.pkl","wb") as f:
        pickle.dump((batch_size, mnist_epochs, num_layers, dimensions), f)

    # load privious representation record
    if Retrain == False and os.path.exists("mnist_net.pkl"):
        print("Loading MNIST model...")
        mnist_net = torch.load("mnist_net.pkl")
        with open("all_representation.pkl", "rb") as f:
            load_all_repre, load_label_y = pickle.load(f)

        return mnist_net, load_all_repre, load_label_y, dimensions, None


    logger.info(f"Training Device : {device}")
    trainLoader, testLoader = utils.dataLoader(batch_size = batch_size)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()

    if opt == "sgd":
        optimizer = torch.optim.SGD(mnist_net.parameters(), lr = lr, momentum = 0.01)
    elif opt == "adam":
        optimizer = torch.optim.Adam(mnist_net.parameters(), lr = lr)

    acc = list()
    # Training
    for epoch in range(mnist_epochs):
        
        running_loss = 0
        
        for i, data in enumerate(trainLoader, 0):
            # 輸入資料
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # 使用 view() 將 inputs 的維度壓到符合模型的輸入。
            if model_type != "cnn":
                inputs = inputs.view(inputs.shape[0], -1) 
        
            # 梯度清空
            optimizer.zero_grad()

            # Forward 
            repre = list(mnist_net(inputs))
            # t1, t2, outputs = mnist_net(inputs)
            
            outputs = None
            # layer transformation
            for idx in range(len(repre)):

                if idx == len(repre)-1: # the last representation
                    outputs = repre[idx]

                if idx in conv_idx and model_type == "cnn": # this layer is convolutional layer
                    repre[idx] = repre[idx].view(-1, len(repre[idx][0])*len(repre[idx][0][0])*len(repre[idx][0][0][0]))
                    repre[idx] = repre[idx].cpu().detach().numpy()
                else:   # ordinary MLP
                    repre[idx] = repre[idx].cpu().detach().numpy()

            labels_np = labels.cpu().detach().numpy()
            
            # transform label to one-hot encoding and save it.
            label_y[epoch] = np.concatenate((label_y[epoch], np.eye(10)[labels_np]),
                                            axis = 0)
            

            # store all representations to additional list
            for layer_idx in range(num_layers):
                all_repre[layer_idx][epoch] = np.concatenate((all_repre[layer_idx][epoch], repre[layer_idx]), axis = 0)

            # backward
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 更新參數
            optimizer.step()
            
            running_loss += loss.item()

        
        logger.info(f"MNIST Training, epoch-{epoch} elapsed time: {time.time()-time1}")
        acc.append(mnist_testing(mnist_net, batch_size, model_type = model_type))
        
    if Retrain == True or not os.path.exists("mnist_net.pkl"):
        torch.save(mnist_net, "mnist_net.pkl")
        with open("all_representation.pkl", "wb") as f:
            pickle.dump((all_repre, label_y), f)
    
    return mnist_net, all_repre, label_y, dimensions, acc

def mnist_testing(mnist_net, batch_size, model_type="mlprelu"):
    # Test
    correct = 0
    total = 0
    device = utils.training_device()
    _, testLoader = utils.dataLoader(batch_size = batch_size)
    with torch.no_grad():
        for data in testLoader:
            inputs, labels = data[0].to(device), data[1].to(device)

            if model_type != "cnn": # need to change shape
                inputs = inputs.view(inputs.shape[0], -1)
            
            # modify if you change MNIST layers structure
            # _, _, outputs = mnist_net(inputs)
            repre = list(mnist_net(inputs))
            outputs = repre[-1]
            
            _, predicted = torch.max(outputs.data, 1) # 找出分數最高的對應channel，即 top-1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the MNIST network on the 10000 test images: %d %%\n' % (100*correct / total))
    
    return 100*correct/total

if __name__ == "__main__":
    mnist_net, _, _ = mnist_training(Retrain=True)
    mnist_testing(mnist_net)
