"""Main Figure 4; saved data2vec summaries, training deferred."""
from common import *
import pickle
import pandas as pd
from fit_utils import plot_insets_pstar_sweep_and_fit

with open(DATA / "data2vec.pkl", "rb") as f:
    per_m = pickle.load(f)

V, L, EMA = 16, 4, 0.99
SEEDS = [0, 1, 2]
MS = [3, 4, 5, 6, 7]
B = 512
P_MIN = 1e6

normalized_series = []
for m in per_m.keys():
    steps, mlp_stack, mlp_avg = per_m[m]
    for seed_idx, seed in enumerate(SEEDS):
        for P, acc in zip(steps * B, mlp_stack[seed_idx]):
            normalized_series.append(
                pd.Series(
                    {
                        "m": m,
                        "P": P,
                        "acc": acc,
                        "seed": seed,
                        "chance_acc": 1 / V,
                    }
                )
            )

normalized_df = pd.DataFrame(normalized_series)

m_values = sorted(per_m.keys())
m_colors = [get_color_sequential(m, m_values[::-1]) for m in m_values]
collapse = lambda steps, M: steps * B / (V * M**3)
raw_P = lambda steps, M: steps * B

def _draw(ax, rescale, fontsize_ticks=None):
    for color, (m, (Ps, stacks, means)) in zip(m_colors, per_m.items()):
        x = rescale(Ps, m)
        stacks = stacks.T
        for xi, seeds in zip(x, stacks):
            ax.plot(
                [xi] * len(seeds), seeds, "o", color=color, alpha=0.35, markersize=4
            )
        ax.plot(x, means, "-", color=color, label=rf"$m = {m}$")
    ax.set_xscale("log")
    if fontsize_ticks is not None:
        ax.tick_params(axis="both", which="both", labelsize=fontsize_ticks)

fig, ax = plt.subplots(1, 1, figsize=half_width_size)
inset_raw = ax.inset_axes([0.04, 0.55, 0.40, 0.42])
inset_pstar = ax.inset_axes([0.68, 0.12, 0.30, 0.40])

fitted_df = plot_insets_pstar_sweep_and_fit(
    inset_raw, inset_pstar, normalized_df, diagnostics=False
)

p_star_pts = []
for color, M in zip(m_colors, m_values):
    steps, mlp_stack, mlp_avg = per_m[M]
    x_main = collapse(steps, M)
    x_raw = raw_P(steps, M)

    for seed_row in mlp_stack:
        ax.plot(x_main, seed_row, "o", color=color, alpha=0.35, markersize=4)
    ax.plot(x_main, mlp_avg, "-", color=color, label=rf"$m = {M}$")

ax.set_xscale("log")
ax.set_xlabel(r"Rescaled number of samples, $P\,/\,vm^3$")
ax.set_ylabel("Classification accuracy")
ax.set_xlim(right=3e5)

inset_raw.annotate(
    "",
    xy=(0.85, 0.10),
    xytext=(0.2, 0.5),
    xycoords="axes fraction",
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", color="black"),
)
inset_raw.text(
    0.02,
    0.52,
    rf"$m={int(min(m_values))}$",
    transform=inset_raw.transAxes,
    fontsize=INSET_ANNOTATION_SIZE,
    ha="left",
    va="bottom",
)
inset_raw.text(
    0.98,
    0.2,
    rf"$m={int(max(m_values))}$",
    transform=inset_raw.transAxes,
    fontsize=INSET_ANNOTATION_SIZE,
    ha="right",
    va="bottom",
)

inset_raw.set_xscale("log")
inset_raw.set_xlim(right=3e8)
inset_raw.set_xlabel(r"Num. of samples, $P$", fontsize=INSET_LABELSIZE, labelpad=1)
inset_raw.tick_params(axis="both", which="both", labelsize=INSET_TICKSIZE, pad=1)
inset_raw.set_yticklabels([])
inset_raw.tick_params(axis="y", which="both", left=False, right=False)

inset_pstar.set_xscale("log")
inset_pstar.set_yscale("log")
inset_pstar.set_xlabel(r"$m$", fontsize=INSET_LABELSIZE, labelpad=1)
inset_pstar.set_ylabel(r"$P^\ast$", fontsize=INSET_LABELSIZE, labelpad=1)
inset_pstar.tick_params(axis="both", which="both", labelsize=INSET_TICKSIZE, pad=1)
inset_pstar.minorticks_off()
inset_pstar.set_xticks(m_values, [str(m) for m in m_values])

p_arr = fitted_df["logP0_mean"].apply(np.exp).to_numpy()
m_arr = np.array(fitted_df["m"].to_numpy(), dtype=float)
ms_ref = m_arr

c_black = np.exp(np.mean(np.log(p_arr) - np.log(V * m_arr**3)))
inset_pstar.plot(ms_ref, c_black * V * ms_ref**3, "k--", linewidth=1.2, zorder=-5)
inset_pstar.annotate(
    r"$\propto m^3$",
    xy=(0.05, 0.85),
    xycoords="axes fraction",
    fontsize=INSET_ANNOTATION_SIZE,
    color="black",
)

m0 = m_arr.min()
p0 = p_arr[m_arr.argmin()]
inset_pstar.plot(ms_ref, p0 * (ms_ref / m0) ** (L + 1), "r--", linewidth=1.2, zorder=-5)
inset_pstar.annotate(
    rf"$\propto m^{{L+1}}$",
    xy=(0.05, 0.70),
    xycoords="axes fraction",
    fontsize=INSET_ANNOTATION_SIZE,
    color="red",
)

handles, labels = ax.get_legend_handles_labels()

fig.subplots_adjust(left=0.13, right=0.99, top=0.95, bottom=0.20)
savefig(fig, "d2vec_main_online_with_pstar")
