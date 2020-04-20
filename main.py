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


if __name__ == "__main__":
    args = get_parser()
    logger = Create_Logger(__name__)

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
    model_type = args.model_type

    if retrain == True:

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
            print(f"del directory : repre")


        logger.info(f"args1: mnist_epochs={mnist_epochs}, batch_size={batch_size}, retrain={retrain}")
        logger.info(f"args2: mnist_lr = {lr}, MNIST optimizer = {opt}, MNIST model_type={model_type}")
        logger.info(f"args3: clean old files = {args.clean_old_files}, folder name = {folder_name}, batch group={batch_group}")
        logger.info(f"args4: show={show}, noise_var={noise_var}, n_epoch={n_epoch}, aamine_epoch={aan_epoch}")

        # train MNIST model
        mnist_net, all_repre, label_y, dimensions, acc = mnist_training(batch_size = batch_size, 
                            mnist_epochs = mnist_epochs, Retrain = args.retrain,
                            lr = lr, opt = opt, model_type = model_type)


        # load MNIST model hyper-parameters config
        with open("mnist_net_config.pkl","rb") as f:
            _, mnist_epochs, num_layers, _ = pickle.load(f)
        with open("repre/label_y.pkl", "wb") as f:
            pickle.dump(label_y, f)

        title = "ip_bs" + str(batch_size) + "_e" + str(mnist_epochs) + "_var" + str(noise_var) \
        + "_bg" + str(batch_group) + "_" + opt + "_lr" + str(lr) + "_mie" + str(n_epoch) + \
        "_amie" + str(aan_epoch) + "_type" + model_type
        for dimension in dimensions:
            title = title + str(dimension) + "_"

        plot_line(acc, title, folder_name)
    # ------------------------------------------------------------

        # seperate batches into batch group
        logger.info(f"seperate batches in the same training epoch in order to calculate MI more individully. ")
        split_all_repre = []
        samples_size = batch_size * batch_group 

        for l_idx in range(num_layers):
            split_all_repre.append([])

            for e_idx in range(mnist_epochs):
                split_all_repre[l_idx].append([])

                for batchGroup_idx in range((60000 // samples_size) + 1):

                    if batchGroup_idx == (60000 // samples_size):
                        try:
                            split_all_repre[l_idx][e_idx].append(all_repre[l_idx][e_idx][samples_size*batchGroup_idx:, ])
                            repre_title = f"repre/layer{l_idx}_epoch{e_idx}_batchGroup{batchGroup_idx}.pkl"
                            with open(repre_title, "wb") as f:
                                pickle.dump(all_repre[l_idx][e_idx][samples_size*batchGroup_idx:, ], f)
                        except: # May occur IndexError or something
                            logger.error("last iteration of seperating batch group.", l_idx, e_idx, batchGroup_idx)
                            logger.error("Errors may caused by mnist training epochs setting mismatch.")
                            exit(0)

                    else:
                        try:
                            split_all_repre[l_idx][e_idx].append(all_repre[l_idx][e_idx][samples_size * batchGroup_idx:samples_size*(batchGroup_idx + 1), ])
                            repre_title = f"repre/layer{l_idx}_epoch{e_idx}_batchGroup{batchGroup_idx}.pkl"
                            with open(repre_title, "wb") as f:
                                pickle.dump(all_repre[l_idx][e_idx][samples_size * batchGroup_idx : samples_size*(batchGroup_idx + 1), ], f)
                        except:
                            print(l_idx, e_idx, samples_size*(batchGroup_idx + 1))
                            exit(0)
        
        # store arguments
        with open("repre/arguments.pkl", "wb") as f:
            pickle.dump((args, title, len(split_all_repre[0][0]), samples_size, dimensions), f)

        del all_repre
    

    with open("repre/label_y.pkl", "rb") as f:
        label_y = pickle.load(f)
    with open("repre/arguments.pkl", "rb") as f:
        args, title, num_batchGroups, samples_size, dimensions = pickle.load(f)

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
    num_layers = len(dimensions)

    all_mi_label = [[] for i in range(num_layers)]
    all_mi_input = [[] for i in range(num_layers)]
    time1 = time.time()

    # use AA-MINE for estimating Information Bottleneck
    for layer_idx in range(num_layers):
        
        # To get better convergence value, we need to adjus MINE model training epochs for different layers
        if layer_idx < 1:
            aan_epoch = args.aamine_epoch + 100
            n_epoch = args.mine_epoch + 100
        else:
            aan_epoch = args.aamine_epoch
            n_epoch = args.mine_epoch
        
        # Adjust noise variance for different layers   2020/04/04
        if layer_idx == 1:
            noise_var = 2
        elif layer_idx == 2:
            noise_var = 2
        else:
            noise_var = args.noise_var

        # for layer representations of each epoch
        for epoch in range(mnist_epochs):

            logger.info(f"== MI == {layer_idx}-th layer, {epoch}-th epoch.")

            # for different batch groups of the same epoch
            for bg_idx in range(num_batchGroups):
                
                mi_flag = "repre/flag_layer" + str(layer_idx) + "epoch" + str(epoch)  + "bg" + str(bg_idx) + ".pkl"

                # This representation have not been calculated
                if not os.path.exists(mi_flag):
                    # save mutual exclusive flag
                    with open(mi_flag, "wb") as f:
                        pickle.dump((layer_idx, epoch, bg_idx), f)

                    # read representation
                    repre_title = f"repre/layer{layer_idx}_epoch{epoch}_batchGroup{bg_idx}.pkl"
                    logger.info(f"Reading file: {repre_title}")
                    with open(repre_title, "rb") as f:
                        repre_t = pickle.load(f)

                    plt.cla() # clear privious plot
                    plt.close("all")
                    # ====================== calculate AA-MINE Mutual Information =============================
                    print("===================================\n")
                    time_stamp = time.time()
                    print(f"AA-MINE -> {layer_idx}-th layer ,{epoch}-th epoch, {bg_idx}-th batch group. \n shape = {repre_t.shape}")
                    try:
                        all_mi_input[layer_idx].append(mi_aamine(repre_t, 
                                input_dim = repre_t.shape[1]*2, noise_var = noise_var, 
                                SHOW=show, n_epoch = aan_epoch, layer_idx = layer_idx, epoch_idx = epoch, batch_idx = bg_idx))
                    except IndexError:
                        print(f"IndexError -> layer_iex/epoch = {layer_idx}/{epoch} ")
                        exit(0)

                    print(f"elapsed time:{time.time()-time_stamp}\n")

                    time_stamp = time.time()
                    # # ====================== calculate MINE Mutual Information =============================
                    print(f"MINE -> {layer_idx}-th layer ,{epoch}-th epoch, {bg_idx}-th batch group. \n shape = {repre_t.shape}/{label_y[epoch][:len(repre_t)].shape}")
                    try:
                        # if not the last batch group
                        if bg_idx != num_batchGroups - 1: 
                            all_mi_label[layer_idx].append(mi_mine(repre_t, label_y[epoch][samples_size * bg_idx:samples_size * (bg_idx + 1)],
                                    input_dim = repre_t.shape[1] + label_y[epoch].shape[1],
                                    SHOW=show, n_epoch = n_epoch, layer_idx = layer_idx, epoch_idx = epoch, batch_idx = bg_idx, 
                                    folder = folder_name))
                        # deal with the last batch group, which is not full samples size
                        else:
                            all_mi_label[layer_idx].append(mi_mine(repre_t, label_y[epoch][samples_size * bg_idx: ],
                                    input_dim = repre_t.shape[1] + label_y[epoch].shape[1],
                                    SHOW=show, n_epoch = n_epoch, layer_idx = layer_idx, epoch_idx = epoch, batch_idx = bg_idx, 
                                    folder = folder_name))
                    except IndexError:
                        print(f"IndexError -> layer_iex/epoch = {layer_idx}/{epoch} ")
                    
                    print(f"elapsed time:{time.time()-time_stamp}\n")

                    # for Debugging : prevent MINE loss from being NaN
                    if math.isnan(all_mi_input[layer_idx][-1]):
                        logger.error(f"AAMINE Mutual Inforamtion is NaN!!!")
                        exit(0)
                    if math.isnan(all_mi_label[layer_idx][-1]):
                        logger.error(f"Mutual information is NaN!!!")
                        print(repre_t)
                        print(label_y[epoch][samples_size * bg_idx:samples_size * (bg_idx + 1)])
                        print(all_mi_label[layer_idx][-1])
                        exit(0)

                    # save MI result, but MINE training image would be saved in "folder"
                    done_mi_file = folder_name + "/mi_layer" + str(layer_idx) + "epoch" + str(epoch) + "bg" + str(bg_idx) +"_done" + ".pkl"
                    with open(done_mi_file, "wb") as f:
                        print(f"layer-{layer_idx}, epoch-{epoch} -> I(T;X) = {all_mi_input[layer_idx][-1]}, I(T;Y) = {all_mi_label[layer_idx][-1]}")
                        pickle.dump((all_mi_input[layer_idx][-1], all_mi_label[layer_idx][-1]), f)
                else:
                    print(f"--- SKIP layer-{layer_idx}, epoch-{epoch}, batchGroup-{bg_idx} ---")


                
        
        # we Plot a information plane after MI of each layer is completed  
        plt.cla() # clear privious plot
        plt.close("all")

        logger.info(f"image title = {title}\n")
        plot_information_plane(all_mi_input, all_mi_label, num_layers, title = title, save = folder_name)

    # logger.info(f"MNIST accuracy = {acc}")
    logger.info(f"Total elapsed time = {time.time()-time1}")

    logger.info("PART 1 is Over")
    exit(0)

    # # ================================================  original PART  ===================================================
    # # print(len(split_all_repre[l_idx][e_idx]),len(split_all_repre[l_idx][e_idx][0]))
    # del all_repre
    
    # all_mi_label = [[] for i in range(num_layers)]
    # all_mi_input = [[] for i in range(num_layers)]
    # time1 = time.time()

    # # use AA-MINE for estimating Information Bottleneck
    # for layer_idx in range(num_layers):
        
    #     # To get better convergence value, we need to adjus MINE model training epochs for different layers
    #     if layer_idx < 1:
    #         aan_epoch = args.mine_epoch + 100
    #         n_epoch = args.mine_epoch + 20
    #     else:
    #         aan_epoch = args.mine_epoch
    #         n_epoch = args.mine_epoch
        
    #     # Adjust noise variance for different layers   2020/04/04
    #     if layer_idx == 1:
    #         noise_var = 1
    #     elif layer_idx == 2:
    #         noise_var = 2.5
    #     else:
    #         noise_var = args.noise_var

    #     # for layer representations of each epoch
    #     for epoch in range(mnist_epochs):

    #         logger.info(f"== MI == {layer_idx}-th layer, {epoch}-th epoch.")

    #         # for different batch groups of the same epoch
    #         for bg_idx in range(len(split_all_repre[l_idx][e_idx])):

    #             plt.cla() # clear privious plot
    #             plt.close("all")
    #             # calculate AA-MINE Mutual Information
    #             print("===================================\n")
    #             time_stamp = time.time()
    #             print(f"AA-MINE -> {layer_idx}-th layer ,{epoch}-th epoch, {bg_idx}-th batch group. \n shape = {repre_t.shape}")
    #             try:
    #                 all_mi_input[layer_idx].append(mi_aamine(repre_t, 
    #                         input_dim = repre_t.shape[1]*2, noise_var = noise_var, 
    #                         SHOW=show, n_epoch = aan_epoch, layer_idx = layer_idx, epoch_idx = epoch, batch_idx = bg_idx))
    #             except IndexError:
    #                 print(f"IndexError -> layer_iex/epoch = {layer_idx}/{epoch} ")
    #                 exit(0)

    #             print(f"elapsed time:{time.time()-time_stamp}\n")

    #             time_stamp = time.time()
    #             # calculate MINE Mutual Information
    #             print(f"MINE -> {layer_idx}-th layer ,{epoch}-th epoch, {bg_idx}-th batch group. \n shape = {repre_t.shape}/{label_y[epoch][:len(repre_t)].shape}")
    #             try:
    #                 # if not the last batch group
    #                 if bg_idx != len(split_all_repre[l_idx][e_idx]) - 1: 
    #                     all_mi_label[layer_idx].append(mi_mine(repre_t, label_y[epoch][samples_size * bg_idx:samples_size * (bg_idx + 1)],
    #                             input_dim = repre_t.shape[1] + label_y[epoch].shape[1],
    #                             SHOW=show, n_epoch = n_epoch, layer_idx = layer_idx, epoch_idx = epoch, batch_idx = bg_idx, 
    #                             folder = folder_name))
    #                 # deal with the last batch group, which is not full samples size
    #                 else:
    #                     all_mi_label[layer_idx].append(mi_mine(repre_t, label_y[epoch][samples_size * bg_idx: ],
    #                             input_dim = repre_t.shape[1] + label_y[epoch].shape[1],
    #                             SHOW=show, n_epoch = n_epoch, layer_idx = layer_idx, epoch_idx = epoch, batch_idx = bg_idx, 
    #                             folder = folder_name))
    #             except IndexError:
    #                 print(f"IndexError -> layer_iex/epoch = {layer_idx}/{epoch} ")
                

    #             # for Debugging : prevent MINE loss from being NaN
    #             if math.isnan(all_mi_input[layer_idx][-1]):
    #                 logger.error(f"AAMINE Mutual Inforamtion is NaN!!!")
    #                 exit(0)
    #             if math.isnan(all_mi_label[layer_idx][-1]):
    #                 logger.error(f"Mutual information is NaN!!!")
    #                 print(repre_t)
    #                 print(label_y[epoch][samples_size * bg_idx:samples_size * (bg_idx + 1)])
    #                 print(all_mi_label[layer_idx][-1])
    #                 exit(0)

    #             print(f"elapsed time:{time.time()-time_stamp}\n")
        
    #     # we Plot a information plane after MI of each layer is completed  
    #     plt.cla() # clear privious plot
    #     plt.close("all")

    #     logger.info(f"image title = {title}\n")
    #     plot_information_plane(all_mi_input, all_mi_label, num_layers, title = title, save = folder_name)

    # logger.info(f"MNIST accuracy = {acc}")
    # logger.info(f"Total elapsed time = {time.time()-time1}")


#    print(all_mi_input, all_mi_label)

