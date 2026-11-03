"""Main Figure 5; saved data2vec summaries, training deferred."""
from common import *
import pickle

D2V_V = 16
D2V_B = 512
D2V_L_RHM = 4
D2V_LEVELS_TO_PLOT = [3, 2, 1]  # rhm_lvl giving ell = 1, 2, 3 left to right

with open(os.path.join(summary_data_path, "data2vec_sensitivity.pkl"), "rb") as f:
    d2v_sensitivity_training = pickle.load(f)

d2v_m_values = sorted(d2v_sensitivity_training.keys())
d2v_m_colors = [get_color_sequential(m, d2v_m_values[::-1]) for m in d2v_m_values]

def _draw_d2v_clustering_score(ax, rhm_lvl, rescale, fontsize_ticks=None):
    for color, (m, (steps, sens_stack, sens_avg)) in zip(
        d2v_m_colors, d2v_sensitivity_training.items()
    ):
        x = rescale(steps, m)
        for seed_row in sens_stack:
            ax.plot(
                x,
                1 - seed_row[:, rhm_lvl],
                "o",
                color=color,
                alpha=0.35,
                markersize=4,
            )
        ax.plot(x, 1 - sens_avg[:, rhm_lvl], "-", color=color, label=rf"$m = {m}$")
    ax.set_xscale("log")
    if fontsize_ticks is not None:
        ax.tick_params(axis="both", which="both", labelsize=fontsize_ticks)

d2v_collapse = lambda steps, m: steps * D2V_B / (D2V_V * m**3)
d2v_raw_p = lambda steps, m: steps * D2V_B

fig, axes = plt.subplots(1, len(D2V_LEVELS_TO_PLOT), figsize=(full_width_size[0], 2.6))
if len(D2V_LEVELS_TO_PLOT) == 1:
    axes = [axes]

axes[0].set_ylabel("Synonym clustering score")
for ax, lvl in zip(axes, D2V_LEVELS_TO_PLOT):
    _draw_d2v_clustering_score(ax, lvl, rescale=d2v_collapse)
    ax.set_xlabel(r"$P\,/\,(v\,m^3)$")
    ax.set_title(rf"$\ell = {D2V_L_RHM - lvl}$")

    inset = ax.inset_axes([0.04, 0.55, 0.40, 0.42])
    _draw_d2v_clustering_score(
        inset, lvl, rescale=d2v_raw_p, fontsize_ticks=INSET_TICKSIZE
    )
    inset.set_xlabel(r"$P$", fontsize=INSET_LABELSIZE, labelpad=1)
    inset.tick_params(axis="both", which="both", pad=1)
    inset.set_yticklabels([])
    inset.tick_params(axis="y", which="both", left=False, right=False)
axes[2].set_yticks([0.0, 0.1])

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=len(d2v_m_values),
    frameon=False,
)
fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.20, wspace=0.30)
savefig(fig, "d2vec_clustering_scores")
plt.close("all")
