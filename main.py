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

from training import mnist_training, mnist_testing
from mine_training import mi_aamine, mi_mine
from plots import plot_information_plane
from utils import get_parser, Create_Logger

if __name__ == "__main__":
    args = get_parser()
    logger = Create_Logger(__name__)

    # arguments passing
    mnist_epochs = args.mnist_epoch
    batch_size = args.batch_size
    retrain = args.retrain
    batch_group = args.batch_group

    show = args.show
    noise_var = args.noise_var
    n_epoch = args.mine_epoch

    logger.info(f"args1: mnist_epochs={mnist_epochs}, batch_size={batch_size}, retrain={retrain}")
    logger.info(f"args2: show={show}, noise_var={noise_var}, n_epoch={n_epoch}")

    # train MNIST model
    mnist_net, all_repre, label_y = mnist_training(batch_size = batch_size, 
                        mnist_epochs = mnist_epochs, Retrain = args.retrain)

    mnist_testing(mnist_net, batch_size)
    
    # load MNIST model hyper-parameters config
    with open("mnist_net_config.pkl","rb") as f:
        _, mnist_epochs, num_layers, _ = pickle.load(f)

# ------------------------------------------------------------


    logger.info(f"seperate batches in the same training epoch in order to calculate MI more individully. ")
    split_all_repre = []
    samples_size = batch_size * batch_group 

    for l_idx in range(num_layers):
        split_all_repre.append([])

        for e_idx in range(mnist_epochs):
            split_all_repre[l_idx].append([])

            for batch_idx in range((60000 // samples_size) + 1):

                if batch_idx == (60000 // samples_size):
                    split_all_repre[l_idx][e_idx].append(all_repre[l_idx][e_idx][samples_size*batch_idx:, ])
                else:
                    try:
                        split_all_repre[l_idx][e_idx].append(all_repre[l_idx][e_idx][samples_size * batch_idx:samples_size*(batch_idx + 1), ])
                        print(all_repre[l_idx][e_idx][samples_size * batch_idx:samples_size*(batch_idx + 1), ].shape)
                    except:
                        print(l_idx, e_idx, samples_size*(batch_idx + 1))
                        break
    
    # print(len(split_all_repre[l_idx][e_idx]),len(split_all_repre[l_idx][e_idx][0]))

    all_mi_label = [[] for i in range(num_layers)]
    all_mi_input = [[] for i in range(num_layers)]
    time1 = time.time()

    # use AA-MINE for estimating Information Bottleneck
    for layer_idx in range(num_layers):
        
        # To get better convergence value, we need to adjus MINE model training epochs for different layers
        if layer_idx < 1:
            n_epoch = args.mine_epoch + 40
        else:
            n_epoch = args.mine_epoch

        # for layer representations of each epoch
        for epoch in range(mnist_epochs):
            
            # for different batch groups of the same epoch
            for bg_idx in range(len(split_all_repre[l_idx][e_idx])):

                plt.cla() # clear privious plot
                plt.close("all")
                
                print("===================================\n")
                time_stamp = time.time()
                print(f"AA-MINE -> {layer_idx}-th layer ,{epoch}-th epoch, {bg_idx}-th batch group. \n shape = {split_all_repre[layer_idx][epoch][bg_idx].shape}")
                try:
                    all_mi_input[layer_idx].append(mi_aamine(split_all_repre[layer_idx][epoch][bg_idx], 
                            input_dim = split_all_repre[layer_idx][epoch][bg_idx].shape[1]*2, noise_var = noise_var, 
                            SHOW=show, n_epoch = n_epoch, layer_idx = layer_idx, epoch_idx = epoch))
                except IndexError:
                    print(f"IndexError -> layer_iex/epoch = {layer_idx}/{epoch} ")
                    exit(0)
                # except RuntimeError:
                #     print(f"== RuntimeError == input_dim = {split_all_repre[layer_idx][epoch][bg_idx].shape[1]*2}")
                #     exit(0)
                print(f"elapsed time:{time.time()-time_stamp}\n")

                time_stamp = time.time()
                print(f"MINE -> {layer_idx}-th layer ,{epoch}-th epoch, {bg_idx}-th batch group. \n shape = {split_all_repre[layer_idx][epoch][bg_idx].shape}/{label_y[epoch].shape}")
                try:
                    all_mi_label[layer_idx].append(mi_mine(split_all_repre[layer_idx][epoch][bg_idx], label_y[epoch],
                            input_dim = split_all_repre[layer_idx][epoch][bg_idx].shape[1] + label_y[epoch].shape[1],
                            SHOW=show, n_epoch = n_epoch, layer_idx = layer_idx, epoch_idx = epoch))
                except IndexError:
                    print(f"IndexError -> layer_iex/epoch = {layer_idx}/{epoch} ")
                print(f"elapsed time:{time.time()-time_stamp}\n")
                
    plt.cla() # clear privious plot
    plt.close("all")

    title = "ip_bs" + str(batch_size) + "_e" + str(mnist_epochs) + "_var" + str(noise_var) + "_mie" + str(n_epoch) 
    for layer_idx in range(num_layers):
        title = title + "_" + str(split_all_repre[layer_idx][bg_idx][0].shape[1])

    print(f"image title = {title}")

    plot_information_plane(all_mi_input, all_mi_label, num_layers, title = title)



# ----------------------------------------





    exit(0)


    all_mi_label = [[] for i in range(num_layers)]
    all_mi_input = [[] for i in range(num_layers)]
    time1 = time.time()

    # use AA-MINE for estimating Information Bottleneck
    for layer_idx in range(num_layers):
        for epoch in range(mnist_epochs):
            time_stamp = time.time()
            print(f"AA-MINE -> {layer_idx}-th layer representation, in {epoch}-th epoch. \n shape = {all_repre[layer_idx][epoch].shape}")
            try:
                all_mi_input[layer_idx].append(mi_aamine(all_repre[layer_idx][epoch], 
                        input_dim = all_repre[layer_idx][epoch].shape[1]*2, noise_var = noise_var, 
                        SHOW=show, n_epoch = n_epoch, layer_idx = layer_idx, epoch_idx = epoch))
            except IndexError:
                print(f"IndexError -> layer_iex/epoch = {layer_idx}/{epoch} ")
                exit(0)
            # except RuntimeError:
            #     print(f"== RuntimeError == input_dim = {all_repre[layer_idx][epoch].shape[1]*2}")
            #     exit(0)
            print(f"elapsed time:{time.time()-time_stamp}\n")

    # use normal MINE for estimating Information Bottleneck
    for layer_idx in range(num_layers):
        for epoch in range(mnist_epochs):

            time_stamp = time.time()
            print(f"MINE -> {layer_idx}-th layer representation, in {epoch}-th epoch. \n shape = {all_repre[layer_idx][epoch].shape}/{label_y[epoch].shape}")
            try:
                all_mi_label[layer_idx].append(mi_mine(all_repre[layer_idx][epoch], label_y[epoch],
                        input_dim = all_repre[layer_idx][epoch].shape[1] + label_y[epoch].shape[1],
                        SHOW=show, n_epoch = n_epoch, layer_idx = layer_idx, epoch_idx = epoch))
            except IndexError:
                print(f"IndexError -> layer_iex/epoch = {layer_idx}/{epoch} ")

            print(f"elapsed time:{time.time()-time_stamp}\n\n")

    title = "ip_bs" + str(batch_size) + "_e" + str(mnist_epochs) + "_var" + str(noise_var) + "_mie" + str(n_epoch) 
    for layer_idx in range(num_layers):
        title = title + "_" + str(all_repre[layer_idx][0].shape[1])

    print(f"image title = {title}")

    plot_information_plane(all_mi_input, all_mi_label, num_layers, title = title)
