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
from utils import get_parser, Create_Logger

def add_noise(x, var = 0.2):
    return x + np.random.normal(0., np.sqrt(var), [x.shape[0], x.shape[1]])

# define function for calculating MI by AA-MINE
def mi_aamine(representation_t, input_dim = 20, noise_var = 0.5, n_epoch = 120,
                 SHOW=True, layer_idx = -1 , epoch_idx = -1, batch_idx = -1):

    model = AA_MINEnet(input_dim).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    plot_loss = []

    for epoch in range(n_epoch):

        x_sample = representation_t # because AA-MINE needs to concate sub-network before MINE
        y_shuffle = np.random.permutation(x_sample)
        
        # 20200405 modify noise algo
        y_sample = add_noise(x_sample, var = noise_var)
        y_shuffle = add_noise(y_shuffle, var = noise_var)

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
    print(f"noise variance = {noise_var}, AA-MINE MI = {final_mi}")
        
    return final_mi

# define function for calculating MI by AA-MINE
def mi_mine(representation_t, y_label, input_dim=20, noise_var = 0.5, n_epoch = 120,
                 SHOW = True, layer_idx = -1 , epoch_idx = -1, batch_idx = -1, folder="mine"):

    model = MINEnet(input_dim).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    plot_loss = []

    for epoch in range(n_epoch):

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

        plt.legend(loc='upper right')
        title = f"MINE_layer{layer_idx}_epoch{epoch_idx}_bgroup{batch_idx}"
        plt.title(title + "_MI = " + str(final_mi))
        plt.savefig(folder + "/" + title + ".png")

        # plt.show()
        
    print(f"MINE MI = {final_mi}")
    return final_mi

# still under testing
def calculate_MI(repre_t = None, repre_y = None, READ = True, input_dim = 20, noise_var = 0.5, n_epoch = 200,
                layer_idx = -1, epoch_idx = -1, batch_idx = -1, SHOW=True, AAMINE = True, folder="mine"):
    '''
        == still testing ==
        this function could calculate either AA-MINE  or MINE,
        allow not pass in representations. (read files automatically)
    '''
    if READ == True:
        # read representations
        repre_file = "repre/layer" + str(layer_idx) + "epoch" + str(epoch_idx) + ".pkl"
        print(f"Reading representation {repre_file}")

        with open(repre_file, "rb") as f:
            repre_t, repre_y = pickle.load(f)
        # set input dimenstions
        if AAMINE == True:
            input_dim = repre_t.shape[1]*2
        else:
            input_dim = repre_t.shape[1] + repre_y.shape[1]

    if AAMINE == True:
        model = AA_MINEnet(input_dim).cuda()
    else:
        model = MINEnet(input_dim).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    plot_loss = []

    for epoch in range(n_epoch):

        x_sample = repre_t # because AA-MINE needs to concate sub-network before MINE
        
        # 20200405 modify noise algo
        if AAMINE == True:
            y_shuffle = np.random.permutation(x_sample)
            y_sample = add_noise(x_sample, var = noise_var)
            y_shuffle = add_noise(y_shuffle, var = noise_var)
        else:
            y_sample = repre_y
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
    final_mi = np.mean(-plot_y[-35:])

    if AAMINE and SHOW:
        plt.plot(-plot_y, color = "b", label="AA-MINE")
        print(f"noise variance = {noise_var}, AA-MINE MI = {final_mi}")
        
    elif AAMINE == False and SHOW:
        plt.plot(-plot_y,color='r', label="MINE")
        plt.legend(loc='upper right')

        title = f"MINE_layer{layer_idx}_epoch{epoch_idx}_bgroup{batch_idx}"
        plt.title(title + "_MI = " + str(final_mi))
        if not os.path.exists(folder):
            os.mkdir(folder)
        plt.savefig(folder + "/" + title + ".png")

        print(f"MINE MI = {final_mi}")
    
    return final_mi
        


if __name__ == "__main__":
    with open("repre/args.pkl", "rb") as f:
        args = pickle.load(f)

    layer_num = args["num_layers"]
    mnist_epoch = args["mnist_epochs"]
    ip_title = args["ip_title"]
    save = args["save"]
    noise_var = args["noise_var"]
    aamine_epoch = args["aamine_epoch"]
    mine_epoch = args["mine_epoch"]
    print("arguments :\n ", args)

    for layer_idx in range(layer_num):
        for epoch in range(mnist_epoch):

            mi_flag = "repre/flag_layer" + str(layer_idx) + "epoch" + str(epoch) + ".pkl"
            # This representation have not been calculated
            if not os.path.exists(mi_flag):
                # save mutual exclusive flag
                with open(mi_flag, "wb") as f:
                    pickle.dump((layer_idx, epoch), f, protocol=4)

                # set noise variance
                if layer_idx == 1:
                    noise_var = 1
                elif layer_idx == 2:
                    noise_var = 2.5
                elif layer_idx == 3:
                    noise_var = 3.5
                else:
                    noise_var = args["noise_var"]

                # AA-MINE
                time_stamp = time.time()
                print("=================================")
                print(f"AA-MINE -> {layer_idx}-th layer ,{epoch}-th epoch")
                mi_x = calculate_MI(READ = True, noise_var = noise_var, n_epoch = aamine_epoch,
                    layer_idx = layer_idx, epoch_idx = epoch, AAMINE = True, folder = save)
                print(f"elapsed time:{time.time()-time_stamp}\n")
                
                # MINE
                time_stamp = time.time()
                print(f"MINE -> {layer_idx}-th layer ,{epoch}-th epoch")
                mi_y = calculate_MI(READ = True, noise_var = noise_var, n_epoch = mine_epoch,
                    layer_idx = layer_idx, epoch_idx = epoch, AAMINE = False, folder = save)
                print(f"elapsed time:{time.time()-time_stamp}\n")

                # save MI result, but MINE training image would be saved in "folder"
                done_mi_file = "repre/mi_layer" + str(layer_idx) + "epoch" + str(epoch) + "_done" + ".pkl"
                with open(done_mi_file, "wb") as f:
                    pickle.dump((mi_x, mi_y), f, protocol=4)
            else:
                print(f"--- SKIP layer-{layer_idx}, epoch-{epoch} ---")
            
