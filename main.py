########################
# 2020/03/28 20:53
# 1. add arguments: --lr, --optimizer, --folder
# 2. add color bar label
# 3. MNIST model training with different optimizer and learning rate

########################





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
import shutil
import math

from training import mnist_training, mnist_testing
from mine_training import mi_aamine, mi_mine
from plots import plot_information_plane, plot_line
from utils import get_parser, Create_Logger


logger = Create_Logger(__name__)

def splitBatchGroup(all_repre, batch_size, batch_group):
    '''
    This Function need to be modify because we don't put all representation in the same file while using CNN
    '''
    logger.info(f"seperate batches in the same training epoch in order to calculate MI more individully. ")
    split_all_repre = []
    samples_size = batch_size * batch_group 

    for l_idx in range(num_layers):
        split_all_repre.append([])

        for e_idx in range(mnist_epochs):
            
            split_all_repre[l_idx].append([])

            for batch_idx in range((60000 // samples_size) + 1):

                if batch_idx == (60000 // samples_size):
                    try:
                        split_all_repre[l_idx][e_idx].append(all_repre[l_idx][e_idx][samples_size*batch_idx:, ])
                    except: # May occur IndexError or something
                        logger.error("last iteration of seperating batch group.", l_idx, e_idx, batch_idx)
                        logger.error("Errors may caused by mnist training epochs setting mismatch.")
                        exit(0)

                else:
                    try:
                        split_all_repre[l_idx][e_idx].append(all_repre[l_idx][e_idx][samples_size * batch_idx:samples_size*(batch_idx + 1), ])
                    except:
                        print(l_idx, e_idx, samples_size*(batch_idx + 1))
                        exit(0)
    
    return split_all_repre 


if __name__ == "__main__":
    args = get_parser()

    # arguments passing
    mnist_epochs = args.mnist_epoch
    batch_size = args.batch_size
    retrain = args.retrain
    batch_group = args.batch_group
    opt = args.mnist_opt
    lr = args.mnist_lr

    show = args.show
    noise_var = args.noise_var
    n_epoch = args.mine_epoch
    aan_epoch = args.aamine_epoch
    folder_name = args.folder_name
    model = args.model_type

    if args.clean_old_files:
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name, ignore_errors=True)
            print(f"del directory : {folder_name}")
        if show:
            os.mkdir(folder_name)
    if not os.path.exists("repre"):
        os.mkdir("repre")
    else:
        shutil.rmtree("repre", ignore_errors=True)
        os.mkdir("repre")


    logger.info(f"args1: mnist_epochs={mnist_epochs}, batch_size={batch_size}, retrain={retrain}")
    logger.info(f"args2: mnist_lr = {lr}, MNIST optimizer = {opt}, model type = {model}")
    logger.info(f"args3: clean old files = {args.clean_old_files}, folder name = {folder_name}, batch group={batch_group}")
    logger.info(f"args4: show={show}, noise_var={noise_var}, n_epoch={n_epoch}, aamine_epoch={aan_epoch}")

    # the name of saved image (information plane)
    ip_title = "ip_bs" + str(batch_size) + "_e" + str(mnist_epochs) + "_var" + str(noise_var) \
            + "_bg" + str(batch_group) + "_" + opt + "_lr" + str(lr) + "_mie" + str(n_epoch) + \
            "_amie" + str(aan_epoch) + "_type" + model

    
    # train MNIST model
    mnist_net, all_repre, label_y, acc = mnist_training(batch_size = batch_size, 
                        mnist_epochs = mnist_epochs, Retrain = args.retrain,
                         lr = lr, opt = opt, model = model)

    # load MNIST model hyper-parameters config
    with open("mnist_net_config.pkl","rb") as f:
        _, mnist_epochs, num_layers, dimensions = pickle.load(f)

    args_dict = {"mnist_epochs":mnist_epochs, "num_layers":num_layers, "ip_title":ip_title,
         "save":folder_name, "noise_var":noise_var, "aamine_epoch":aan_epoch, "mine_epoch":n_epoch}
    with open("repre/args.pkl", "wb") as f:        
        pickle.dump(args_dict, f)
    
    for layer_idx in range(num_layers):
        ip_title = ip_title + "_" + str(dimensions[layer_idx])

    logger.info(f"MNIST training Finished !!")
    plot_line(acc, ip_title, folder_name)
# ------------------------------------------------------------


    # logger.info(f"seperate batches in the same training epoch in order to calculate MI more individully. ")
    # split_all_repre = []
    # samples_size = batch_size * batch_group 

    # for l_idx in range(num_layers):
    #     split_all_repre.append([])

    #     for e_idx in range(mnist_epochs):
            
    #         split_all_repre[l_idx].append([])

    #         for batch_idx in range((60000 // samples_size) + 1):

    #             if batch_idx == (60000 // samples_size):
    #                 try:
    #                     split_all_repre[l_idx][e_idx].append(all_repre[l_idx][e_idx][samples_size*batch_idx:, ])
    #                 except: # May occur IndexError or something
    #                     logger.error("last iteration of seperating batch group.", l_idx, e_idx, batch_idx)
    #                     logger.error("Errors may caused by mnist training epochs setting mismatch.")
    #                     exit(0)

    #             else:
    #                 try:
    #                     split_all_repre[l_idx][e_idx].append(all_repre[l_idx][e_idx][samples_size * batch_idx:samples_size*(batch_idx + 1), ])
    #                 except:
    #                     print(l_idx, e_idx, samples_size*(batch_idx + 1))
    #                     exit(0)
                        
# ------------------------------------------------------------
    
    del all_repre
    
    all_mi_label = [[] for i in range(num_layers)]
    all_mi_input = [[] for i in range(num_layers)]
    time1 = time.time()

    # use AA-MINE and MINE for estimating Information Bottleneck
    for layer_idx in range(num_layers):
        
        # To get better convergence value, we need to adjus MINE model training epochs for different layers
        if layer_idx < 1:
            aan_epoch = args.mine_epoch + 100
            n_epoch = args.mine_epoch + 20
        else:
            aan_epoch = args.mine_epoch
            n_epoch = args.mine_epoch

        # Adjust noise variance for different layers   2020/04/04
        if layer_idx == 1:
            noise_var = 1
        elif layer_idx == 2:
            noise_var = 2.5
        elif layer_idx == 3:
            noise_var = 3.5
        else:
            noise_var = args.noise_var

        # for layer representations of each epoch
        for epoch in range(mnist_epochs):
            logger.info(f"== MI == {layer_idx}-th layer, {epoch}-th epoch")

            # for different batch groups of the same epoch
            # for bg_idx in range(len(split_all_repre[l_idx][e_idx])):

            repre_file = "repre/layer" + str(layer_idx) + "epoch" + str(epoch) + ".pkl"
            with open(repre_file, "rb") as f:
                split_all_repre, label_y = pickle.load(f)

            plt.cla() # clear privious plot
            plt.close("all")
            
            print("===================================\n")
            time_stamp = time.time()
            print(f"AA-MINE -> {layer_idx}-th layer ,{epoch}-th epoch,  \n shape = {split_all_repre.shape}")
            try:
                all_mi_input[layer_idx].append(mi_aamine(split_all_repre, 
                        input_dim = split_all_repre.shape[1]*2, noise_var = noise_var, 
                        SHOW=show, n_epoch = aan_epoch, layer_idx = layer_idx, epoch_idx = epoch))
            except IndexError:
                print(f"IndexError -> layer_iex/epoch = {layer_idx}/{epoch} ")
                exit(0)
            print(f"elapsed time:{time.time()-time_stamp}\n")

            time_stamp = time.time()
            print(f"MINE -> {layer_idx}-th layer ,{epoch}-th epoch,  \n shape = {split_all_repre.shape}/{label_y.shape}")
            try:
                all_mi_label[layer_idx].append(mi_mine(split_all_repre, label_y,
                        input_dim = split_all_repre.shape[1] + label_y.shape[1],
                        SHOW=show, n_epoch = n_epoch, layer_idx = layer_idx, epoch_idx = epoch, folder = folder_name))

            except IndexError:
                logger.error(f"IndexError -> layer_iex/epoch = {layer_idx}/{epoch} ")
            

            # for Debugging : prevent MINE loss from being NaN
            if math.isnan(all_mi_input[layer_idx][-1]):
                logger.error(f"AAMINE Mutual Inforamtion is NaN!!!")
                exit(0)
            if math.isnan(all_mi_label[layer_idx][-1]):
                logger.error(f"Mutual information is NaN!!!")
                print(split_all_repre)
                print(label_y)
                print(all_mi_label[layer_idx][-1])
                exit(0)

            print(f"elapsed time:{time.time()-time_stamp}\n")
                
        # we Plot a information plane after MI of each layer is completed    
        plt.cla() # clear privious plot
        plt.close("all")

        logger.info(f"image ip_title = {ip_title}\n")
        plot_information_plane(all_mi_input, all_mi_label, num_layers, title = ip_title, save = folder_name)

    logger.info(f"MNIST accuracy = {acc}")
    logger.info(f"Total elapsed time = {time.time()-time1}")

#    print(all_mi_input, all_mi_label)

