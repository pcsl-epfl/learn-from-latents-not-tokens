import matplotlib.pyplot as plt
import numpy as np

def logx(ax):
    ax.set_xscale("log")

def loglog(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")

def get_color_sequential_colormap(value, values, colormap="plasma", cmap_range=(0, 1)):
    values = list(values)
    fraction = values.index(value) / max(1, len(values) - 1)
    return plt.get_cmap(colormap)(cmap_range[0] + np.diff(cmap_range)[0] * fraction)
