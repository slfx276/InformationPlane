import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
import pickle


cmap = ["viridis", "plasma", "cividis", "magma", "inferno"]

def plot_figures(x, y, axs, cmap_idx = 0):
    '''
        plot a multi-colored line on figure.
    '''
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # print(segments)
    # fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(0, len(x) )

    lc = LineCollection(segments, cmap = cmap[cmap_idx], norm=norm)
    
    # Set the values used for colormapping
    lc.set_array(np.arange(0, len(x), 1))

    lc.set_linewidth(2)
    return lc

def find_value(twoDimList, v_type="min"):
    if v_type == "min":
        try:
            return min([min(layer_mi) for layer_mi in twoDimList ])
        except:
            return 0
    elif v_type == "max":
        try:
            return max([max(layer_mi) for layer_mi in twoDimList ])
        except:
            return 12

def plot_information_plane(mi_x, mi_y, total_layers, title = "ip", save = "mine"):

    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    fig.suptitle(title)
    axs.set_xlim(find_value(mi_x, v_type="min"), find_value(mi_x, v_type="max") + 0.2)
    axs.set_ylim(find_value(mi_y, v_type="min"), find_value(mi_y, v_type="max") + 0.2)
    axs.set_xlabel("I(T;X)")
    axs.set_ylabel("I(T;Y)")

    PlotBar = False
    for layer_idx in range(total_layers):
        PlotBar = True
        lc = plot_figures(mi_x[layer_idx], mi_y[layer_idx], axs, cmap_idx = layer_idx % 5 )
        line = axs.add_collection(lc)
        # fig.colorbar(line, ax=axs)
        # if not PlotBar:
        cbar = fig.colorbar(line, ax=axs)
        cbar.set_label(f"l{layer_idx}") ###############
    # fig.colorbar(line, ax=axs)
    if save != "mine":
        plt.savefig(title + "_" + save + ".png")
        print(f"save information plane image: {title}_{save}.png")
    else:
        plt.savefig(title + ".png")
        print(f"save information plane image: {title}.png")
    plt.show()


def plot_line(value_list, title = "", save = ""):
    plt.cla() # clear privious plot
    plt.close("all")
    plt.plot(value_list)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(title + save)
    plt.savefig("acc_" + title + "_" + save + ".png")
    print(f"save image: acc_{title}_{save}.png")


def combine_mi():
    '''
    read the MI files saved in folder "repre"
    which are calculated by excuting "python mine_training.py ...." (parallel)
    then plot the information plane.
    '''
    with open("repre/args.pkl", "rb") as f:
        args = pickle.load(f)

    num_layers = args["num_layers"]
    mnist_epochs = args["mnist_epochs"]
    ip_title = args["ip_title"]
    save = args["save"]

    mi_tx = [[] for i in range(num_layers)]
    mi_ty = [[] for i in range(num_layers)]
    # read all MI files
    for layer_idx in range(num_layers):
        for epoch in range(mnist_epochs):
            done_mi_file = "repre/mi_layer" + str(layer_idx) + "epoch" + str(epoch) + "_done" + ".pkl"
            with open(done_mi_file, "rb") as f:
                x, y = pickle.load(f)
                mi_tx[layer_idx].append(x)
                mi_ty[layer_idx].append(y)

    plot_information_plane(mi_tx, mi_ty, num_layers, title = ip_title, save = save)
    


if __name__ == "__main__":
    combine_mi()

