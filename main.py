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


if __name__ == "__main__":
    mnist_net, all_repre, label_y = mnist_training(Retrain=True)
    mnist_testing(mnist_net)

    with open("mnist_net_config.pkl","rb") as f:
        _, mnist_epochs, num_layers, _ = pickle.load(f)

    noise_var = 0.2
    n_epoch = 120

    all_mi_label = [[] for i in range(num_layers)]
    all_mi_input = [[] for i in range(num_layers)]
    time1 = time.time()

    # use AA-MINE
    for layer_idx in range(num_layers):
        for epoch in range(mnist_epochs):
            time_stamp = time.time()
            print(f"AA-MINE -> {layer_idx}-th layer representation, in {epoch}-th epoch. \n shape = {all_repre[layer_idx][epoch].shape}")
            try:
                all_mi_input[layer_idx].append(mi_aamine(all_repre[layer_idx][epoch], 
                        input_dim = all_repre[layer_idx][epoch].shape[1]*2, noise_var = noise_var, SHOW=False, n_epoch = n_epoch))
            except IndexError:
                print(f"IndexError -> layer_iex/epoch = {layer_idx}/{epoch} ")
                exit(0)
            # except RuntimeError:
            #     print(f"== RuntimeError == input_dim = {all_repre[layer_idx][epoch].shape[1]*2}")
            #     exit(0)
            print(f"elapsed time:{time.time()-time_stamp}\n")

    # use normal MINE      
    for layer_idx in range(num_layers):
        for epoch in range(mnist_epochs):

            time_stamp = time.time()
            print(f"MINE -> {layer_idx}-th layer representation, in {epoch}-th epoch. \n shape = {all_repre[layer_idx][epoch].shape}/{label_y[epoch].shape}")
            try:
                all_mi_label[layer_idx].append(mi_mine(all_repre[layer_idx][epoch], label_y[epoch],
                        input_dim = all_repre[layer_idx][epoch].shape[1] + label_y[epoch].shape[1], SHOW=False, n_epoch = n_epoch))
            except IndexError:
                print(f"IndexError -> layer_iex/epoch = {layer_idx}/{epoch} ")

            print(f"elapsed time:{time.time()-time_stamp}\n\n")

    # plot name
    plt.figure('information plane')
    ax = plt.gca()
    # set x,y axis name
    ax.set_xlabel('I(T;X)')
    ax.set_ylabel('I(T;Y)')

    # 画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
    # 参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度
    color = ['r','g','b']
    for layer_idx in range(num_layers):
        ax.plot(all_mi_input[layer_idx], all_mi_label[layer_idx], color = color[layer_idx % 3], linewidth=1, alpha=0.6)

    print(f"elapsed time = {time.time() - time1}\nnoise_var = {noise_var}\n{num_layers} layers/{mnist_epochs} epochs\n\n")

    plt.show()
    plt.savefig("information_plane.png")

    plot_information_plane(all_mi_input, all_mi_label, 3)
