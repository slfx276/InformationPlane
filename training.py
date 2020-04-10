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

from model import MLP_relu, MLP_tanh, CNN
import utils



logger = utils.Create_Logger(__name__)

def mnist_training(batch_size, mnist_epochs, Retrain = False, lr = 0.001, 
                                        opt = "adam", model = "mlp"):
    time1 = time.time()

    device = utils.training_device()
    conv_idx = list()
    if model == "mlptanh":
        mnist_net = MLP_tanh().to(device)
        dimensions = [500, 256, 10] # You have to adjust this if you change MNIST model dimension
    elif model == "mlprelu":
        mnist_net = MLP_relu().to(device)
        dimensions = [500, 256, 10] # You have to adjust this if you change MNIST model dimension
    elif model == "cnn":
        mnist_net = CNN().to(device)
        conv_idx = [0, 1] # which layer is convolutional layer
        dimensions = [10*24*24, 20*10*10, 256, 10]
        
    # create storage container of hidden layer representations 
    
    num_layers = len(dimensions)

    label_y = [np.empty(shape=[0, 10]) for i in range(mnist_epochs)]

    # all_repre = []
    # for layer_idx in range(num_layers):
    #     all_repre.append([])
    #     for epoch in range(mnist_epochs):
    #         all_repre[layer_idx].append(np.empty(shape = [0, dimensions[layer_idx]]))

    # save training model config
    with open("mnist_net_config.pkl","wb") as f:
        pickle.dump((batch_size, mnist_epochs, num_layers, dimensions), f)

    # load privious representation record
    # === This loading part need to be modified 20200409 ===
    if Retrain == False and os.path.exists("repre/mnist_net.pkl"):
        print("Loading MNIST model...")
        mnist_net = torch.load("repre/mnist_net.pkl")
        with open("repre/all_representation.pkl", "rb") as f:
            load_all_repre, load_label_y = pickle.load(f)

        return mnist_net, load_all_repre, load_label_y, None


    logger.info(f"Training Device : {device}")
    trainLoader, testLoader = utils.dataLoader(batch_size = batch_size)


    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()

    if opt == "sgd":
        optimizer = torch.optim.SGD(mnist_net.parameters(), lr = lr, momentum = 0.9)
    elif opt == "adam":
        optimizer = torch.optim.Adam(mnist_net.parameters(), lr = lr)

    logger.info(mnist_net)

    acc = list()
    # Training
    for epoch in range(mnist_epochs):

        all_repre = []
        for layer_idx in range(num_layers):
            all_repre.append(np.empty(shape = [0, dimensions[layer_idx]]))

        running_loss = 0
        
        for i, data in enumerate(trainLoader, 0):
            # 輸入資料
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # 使用 view() 將 inputs 的維度壓到符合模型的輸入。
            if model != "cnn":
                inputs = inputs.view(inputs.shape[0], -1) 
        
            # 梯度清空
            optimizer.zero_grad()

            # Forward 
            repre = list(mnist_net(inputs))
            # logger.debug(f"repre dimensions : {len(repre)}, {len(repre[0])}, {len(repre[0][0])}, {len(repre[0][0][0])}")


            # t1, t2, outputs = mnist_net(inputs)
            predict = None
            # layer transformation
            for idx in range(len(repre)):

                if idx == len(repre)-1: # the last representation
                    predict = repre[idx]

                if idx in conv_idx and model == "cnn": # this layer is convolutional layer
                    repre[idx] = repre[idx].view(-1, len(repre[idx][0])*len(repre[idx][0][0])*len(repre[idx][0][0][0]))
                    repre[idx] = repre[idx].cpu().detach().numpy()
                    continue

                repre[idx] = repre[idx].cpu().detach().numpy()
            # t1, t2, outputs_np = t1.cpu().detach().numpy(), t2.cpu().detach().numpy(), \
            #                                         outputs.cpu().detach().numpy()

            labels_np = labels.cpu().detach().numpy()
            
            # transform label to one-hot encoding and save it.
            label_y[epoch] = np.concatenate((label_y[epoch], np.eye(10)[labels_np]),
                                            axis = 0)
            

            # store all representations to additional list
            # for layer_idx in range(num_layers):
            #     all_repre[layer_idx][epoch] = np.concatenate((all_repre[layer_idx][epoch], repre[layer_idx]), axis = 0)
            for layer_idx in range(num_layers):
                all_repre[layer_idx] = np.concatenate((all_repre[layer_idx], repre[layer_idx]), axis = 0)
                    
            # all_repre[0][epoch] = np.concatenate((all_repre[0][epoch], t1), axis = 0)
            # all_repre[1][epoch] = np.concatenate((all_repre[1][epoch], t2), axis = 0)
            # all_repre[2][epoch] = np.concatenate((all_repre[2][epoch], outputs_np), axis = 0)

            # backward
            loss = criterion(predict, labels)
            loss.backward()
            
            # 更新參數
            optimizer.step()
            
            running_loss += loss.item()

            if i % 2000 == 937:
                print('[%d/%d, %d/%d] loss: %.3f' % (epoch+1, mnist_epochs, i+1
                                        , len(trainLoader), running_loss/2000))
                running_loss = 0.0 

        if Retrain == True:
            if not os.path.exists("mnist_net.pkl"):
                torch.save(mnist_net, "repre/mnist_net.pkl")

            # save each layer's representation in this epoch
            for layer_idx in range(num_layers):
                repre_file = "repre/layer" + str(layer_idx) + "epoch" + str(epoch) + ".pkl"
                with open(repre_file, "wb") as f:
                    pickle.dump((all_repre[layer_idx], label_y[epoch]), f, protocol=4)


        logger.info(f"Training epoch {epoch}, elapsed time: {time.time()-time1}")
        acc.append(mnist_testing(mnist_net, batch_size, model = model))
    

        
    # if Retrain == True or not os.path.exists("mnist_net.pkl"):
    #     torch.save(mnist_net, "training_result/mnist_net.pkl")
    #     with open("training_result/all_representation.pkl", "wb") as f:
    #         pickle.dump((all_repre, label_y), f, protocol=4)
    
    return mnist_net, all_repre, label_y, acc

def mnist_testing(mnist_net, batch_size, model="mlprelu"):
    # Test
    correct = 0
    total = 0
    device = utils.training_device()
    _, testLoader = utils.dataLoader(batch_size = batch_size)
    with torch.no_grad():
        for data in testLoader:
            inputs, labels = data[0].to(device), data[1].to(device)
            if model != "cnn": # need to change shape
                inputs = inputs.view(inputs.shape[0], -1)
            
            # modify if you change MNIST layers structure
            outputs = list(mnist_net(inputs))[-1]
            
            _, predicted = torch.max(outputs.data, 1) # 找出分數最高的對應channel，即 top-1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the MNIST network on the 10000 test images: %d %%\n' % (100*correct / total))
    
    return 100*correct/total

if __name__ == "__main__":
    mnist_net, _, _ = mnist_training(Retrain=True)
    mnist_testing(mnist_net)
