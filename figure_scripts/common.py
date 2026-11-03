"""Shared paper typography, paths and PDF/PNG exports."""
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import plot_utils as ph
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("REPRODUCTION_DATA", ROOT / "data/reproduction_data"))
OUTPUT = Path(os.environ.get("FIGURE_OUTPUT", ROOT / "figures"))
OUTPUT.mkdir(parents=True, exist_ok=True)
# Here, we should put rcparam settings for the paper, so that all figures are consistent.
# plt.rcParams.update(
#     {
#         "font.size": 8,
#         "axes.titlesize": 8,
#         "axes.labelsize": 8,
#         "xtick.labelsize": 7,
#         "ytick.labelsize": 7,
#         "legend.fontsize": 7,
#     }
# )
# --- NeurIPS-style typography ---
plt.style.use("tableau-colorblind10")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
    }
)

half_width_size = (4, 3)
full_width_size = (8, 3)
# default colormap for all figures:
cmap = "plasma"


INSET_LABELSIZE = 8
INSET_TICKSIZE = 7
INSET_ANNOTATION_SIZE = 8


def get_color_sequential(value, values, cmap=cmap):
    return ph.get_color_sequential_colormap(
        value, values, colormap=cmap, cmap_range=(0.0, 0.8)
    )



plt.rcParams["text.usetex"] = os.environ.get("USE_TEX", "1") != "0"
pstar_f = lambda v, m: v * m**3
pstar_latex = r"v m^3"
summary_data_path = str(DATA)

def savefig(fig, name):
    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)
