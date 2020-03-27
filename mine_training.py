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

from model import AA_MINEnet, MINEnet 
import utils

def add_noise(x, var = 0.2):
    return x + np.random.normal(0., np.sqrt(var), [x.shape[0], x.shape[1]])

# define function for calculating MI by AA-MINE
def mi_aamine(representation_t, input_dim = 20, noise_var = 0.5, n_epoch = 120,
                 SHOW=True, layer_idx = -1 , epoch_idx = -1, batch_idx = -1):

    model = AA_MINEnet(input_dim).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)
    plot_loss = []

    for epoch in range(n_epoch):
        if epoch%1000 == 0:
            print(f"epoch of AA-MINE= {epoch}")

        x_sample = representation_t # because AA-MINE needs to concate sub-network before MINE
        y_sample = add_noise(x_sample, var = noise_var)
        y_shuffle = np.random.permutation(y_sample)

        x_sample = Variable(torch.from_numpy(x_sample).type(torch.FloatTensor), requires_grad = True).cuda()
        y_sample = Variable(torch.from_numpy(y_sample).type(torch.FloatTensor), requires_grad = True).cuda()
        y_shuffle = Variable(torch.from_numpy(y_shuffle).type(torch.FloatTensor), requires_grad = True).cuda()

        # try:
        pred_xy = model(x_sample, y_sample)
        pred_x_y = model(x_sample, y_shuffle)
        # except:
        #     print("x sample : ", type(x_sample), x_sample.shape)
        #     print("y sample :", type(y_sample), y_sample.shape)
        #     print("y shuffle :", type(y_shuffle), y_shuffle.shape)
        #     exit(0)

        ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        loss = - ret  # maximize
        plot_loss.append(loss.cpu().data.numpy())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        # plot image of MI trend
        plot_x = np.arange(len(plot_loss))
        plot_y = np.array(plot_loss).reshape(-1,)
        if SHOW:
            plt.plot(-plot_y, color = "b", label="AA-MINE")
           
        
    final_mi = np.mean(-plot_y[-35:])
    if SHOW:
        if not os.path.exists("MINE"):
            os.mkdir("MINE")
        # plt.legend(loc='upper right')
        # title = f"AAMINE_layer{layer_idx}_epoch{epoch_idx}_bgroup{batch_idx}"
        # plt.title(title + "_MI = " + str(final_mi))
        # plt.savefig("./MINE/"+title + ".png")
        # plt.show()
        
    print(f"noise variance = {noise_var}, AA-MINE MI = {final_mi}")
    return final_mi

# define function for calculating MI by AA-MINE
def mi_mine(representation_t, y_label, input_dim=20, noise_var = 0.5, n_epoch = 120,
                 SHOW = True, layer_idx = -1 , epoch_idx = -1, batch_idx = -1):

    model = MINEnet(input_dim).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    plot_loss = []

    for epoch in range(n_epoch):
        if epoch%1000 == 0:
            print(f"epoch of MINE= {epoch}")

        x_sample = representation_t
        y_sample = y_label
        y_shuffle = np.random.permutation(y_sample)

        x_sample = Variable(torch.from_numpy(x_sample).type(torch.FloatTensor), requires_grad = True).cuda()
        y_sample = Variable(torch.from_numpy(y_sample).type(torch.FloatTensor), requires_grad = True).cuda()
        y_shuffle = Variable(torch.from_numpy(y_shuffle).type(torch.FloatTensor), requires_grad = True).cuda()

        pred_xy = model(x_sample, y_sample)
        pred_x_y = model(x_sample, y_shuffle)

        ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        loss = - ret  # maximize
        plot_loss.append(loss.cpu().data.numpy())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        # plot image of MI trend
        plot_x = np.arange(len(plot_loss))
        plot_y = np.array(plot_loss).reshape(-1,)
        if SHOW:
            plt.plot(-plot_y,color='r', label="MINE")
    
    final_mi = np.mean(-plot_y[-35:])
    
    if SHOW:
        if not os.path.exists("MINE"):
            os.mkdir("MINE")

        plt.legend(loc='upper right')
        title = f"MINE_layer{layer_idx}_epoch{epoch_idx}_bgroup{batch_idx}"
        plt.title(title + "_MI = " + str(final_mi))
        plt.savefig("./MINE/"+title + ".png")
        # plt.show()
        
    print(f"MINE MI = {final_mi}")
    return final_mi

if __name__ == "__main__":
    with open("all_repre.pkl", "rb") as f:
        all_repre = pickle.load(f)
    print(all_repre)