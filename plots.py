import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap



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
        return min([min(layer_mi) for layer_mi in twoDimList ])
    elif v_type == "max":
        return max([max(layer_mi) for layer_mi in twoDimList ])

def plot_information_plane(mi_x, mi_y, total_layers, title = "ip", save = None):
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
    if save == None:
        plt.savefig(title + ".png")
    else:
        plt.savefig(save + ".png")
    plt.show()


if __name__ == "__main__":
    mi_x = [[1, 2, 5, 6],[7, 5, 8,11], [4, 5, 6, 7], [1, 3, 5, 7]]
    mi_y = [[i for i in range(4)] for j in range(len(mi_x))]
    layer_num = len(mi_x)
    plot_information_plane(mi_x, mi_y, layer_num)


