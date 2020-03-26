import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap

def plot_figures(x, y, axs):
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

    lc = LineCollection(segments, cmap='viridis', norm=norm)

    # Set the values used for colormapping
    lc.set_array(np.arange(0, len(x), 1))

    lc.set_linewidth(2)
    return lc

def find_value(twoDimList, v_type="min"):
    if v_type == "min":
        return min([min(layer_mi) for layer_mi in twoDimList ])
    elif v_type == "max":
        return max([max(layer_mi) for layer_mi in twoDimList ])

def plot_information_plane(mi_x, mi_y, total_layers):
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    axs.set_xlim(find_value(mi_x), find_value(mi_x, v_type="max"))
    axs.set_ylim(find_value(mi_y), find_value(mi_y, v_type="max"))

    PlotBar = False
    for layer_idx in range(total_layers):
        PlotBar = True
        lc = plot_figures(mi_x[layer_idx], mi_y[layer_idx], axs)
        line = axs.add_collection(lc)
        if not PlotBar:
            fig.colorbar(line, ax=axs)

    plt.show()


if __name__ == "__main__":
    layer_num = 2
    mi_x = [[1, 2, 5, 6],[7, 5, 8,11]]
    mi_y = [[i for i in range(4)] for j in range(2)]
    plot_ip(mi_x, mi_y, layer_num)


