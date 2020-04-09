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

from model import MNIST_Net
import utils



logger = utils.Create_Logger(__name__)

def mnist_training(batch_size, mnist_epochs, Retrain = False, lr = 0.001, opt = "sgd"):
    time1 = time.time()


    # create storage container of hidden layer representations 
    dimensions = [500, 256, 10] # You have to adjust this if you change MNIST model dimension
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

        return mnist_net, load_all_repre, load_label_y


    device = utils.training_device()
    logger.info(f"Training Device : {device}")
    trainLoader, testLoader = utils.dataLoader(batch_size = batch_size)
    mnist_net = MNIST_Net().to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()

    if opt == "sgd":
        optimizer = torch.optim.SGD(mnist_net.parameters(), lr = lr, momentum = 0.01)
    elif opt == "adam":
        optimizer = torch.optim.Adam(mnist_net.parameters(), lr = lr)

    # Training
    for epoch in range(mnist_epochs):
        
        running_loss = 0
        
        for i, data in enumerate(trainLoader, 0):
            # 輸入資料
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # 使用 view() 將 inputs 的維度壓到符合模型的輸入。
            inputs = inputs.view(inputs.shape[0], -1) 
        
            # 梯度清空
            optimizer.zero_grad()

            # Forward 
            t1, t2, outputs = mnist_net(inputs)
            
            # layer transformation
            t1, t2, outputs_np = t1.cpu().detach().numpy(), t2.cpu().detach().numpy(), \
                                                    outputs.cpu().detach().numpy()
            inputs_np = inputs.cpu().detach().numpy()
            labels_np = labels.cpu().detach().numpy()
            
            # transform label to one-hot encoding and save it.
            label_y[epoch] = np.concatenate((label_y[epoch], np.eye(10)[labels_np]),
                                            axis = 0)
            

            # store all representations to additional list
            all_repre[0][epoch] = np.concatenate((all_repre[0][epoch], t1), axis = 0)
            all_repre[1][epoch] = np.concatenate((all_repre[1][epoch], t2), axis = 0)
            all_repre[2][epoch] = np.concatenate((all_repre[2][epoch], outputs_np), axis = 0)

            # backward
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 更新參數
            optimizer.step()
            
            running_loss += loss.item()

            if i % 2000 == 937:
                print('[%d/%d, %d/%d] loss: %.3f' % (epoch+1, mnist_epochs, i+1
                                        , len(trainLoader), running_loss/2000))
                running_loss = 0.0 
        
        logger.info(f"Finishe Training, elapsed time: {time.time()-time1}")
        mnist_testing(mnist_net, batch_size)
    
        
    if Retrain == True or not os.path.exists("mnist_net.pkl"):
        torch.save(mnist_net, "mnist_net.pkl")
        with open("all_representation.pkl", "wb") as f:
            pickle.dump((all_repre, label_y), f)
    
    return mnist_net, all_repre, label_y

def mnist_testing(mnist_net, batch_size):
    # Test
    correct = 0
    total = 0
    device = utils.training_device()
    _, testLoader = utils.dataLoader(batch_size = batch_size)
    with torch.no_grad():
        for data in testLoader:
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.view(inputs.shape[0], -1)
            
            # modify if you change MNIST layers structure
            _, _, outputs = mnist_net(inputs)
            
            _, predicted = torch.max(outputs.data, 1) # 找出分數最高的對應channel，即 top-1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the MNIST network on the 10000 test images: %d %%\n' % (100*correct / total))
    
    return 100*correct/total

if __name__ == "__main__":
    mnist_net, _, _ = mnist_training(Retrain=True)
    mnist_testing(mnist_net)
