"""Main Figure 3: ILC and gradient-based ILC sample-complexity sweeps.
Distilled from notebooks/make_all_figs.ipynb cells 20, 22, 24, 26.
Per-seed logistic fits and Monte Carlo uncertainty are recomputed here.
"""
from common import *
import pandas as pd
from fit_utils import plot_insets_pstar_sweep_and_fit
import os
np.random.seed(0)
df_ptc_m = pd.read_parquet(DATA / "df_ptc_m_sweep_best.parquet")
df_e2e_statistician = pd.read_parquet(DATA / "df_e2e_statistician.parquet")

def plot_ptc_m_sweep(ax, m_L_exp: int | None = None, insets: bool = True):
    if insets:
        axins = ax.inset_axes([0.04, 0.6, 0.4, 0.35])
        axins2 = ax.inset_axes([0.65, 0.15, 0.35, 0.43])
        df_subset = df_ptc_m[df_ptc_m["experiment_name"] == "ptc_m_sweep_L4"]
        normalized_df = pd.DataFrame()
        normalized_df["m"] = df_subset["rhm_m"]
        normalized_df["P"] = df_subset["rhm_Ntrain"]
        normalized_df["acc"] = df_subset["classifier_test_acc_at_best_val"]
        normalized_df["seed"] = df_subset["rhm_seed_rules"]
        normalized_df["chance_acc"] = 1 / df_subset["rhm_v"]
        plot_insets_pstar_sweep_and_fit(
            axins, axins2, normalized_df, diagnostics=False, min_fit_m=8
        )

    m_values = sorted(df_ptc_m["rhm_m"].unique())
    v = df_ptc_m["rhm_v"].unique()[0]
    L = df_ptc_m["rhm_L"].unique()[0]
    for m in m_values:
        color = get_color_sequential(m, m_values[::-1])
        df_subset = df_ptc_m[
            (df_ptc_m["experiment_name"] == "ptc_m_sweep_L4") & (df_ptc_m["rhm_m"] == m)
        ].copy()
        pstar = pstar_f(v, m)
        if m_L_exp is not None:
            pstar = df_subset["rhm_v"].iloc[0] * df_subset["rhm_m"].iloc[0] ** m_L_exp

        ax.scatter(
            df_subset["rhm_Ntrain"] / (pstar),
            df_subset["classifier_test_acc_at_best_val"],
            color=color,
            alpha=0.55,
            edgecolor="none",
            label=None,
        )
        mean_df = (
            df_subset.groupby("rhm_Ntrain", as_index=False)[
                "classifier_test_acc_at_best_val"
            ]
            .mean()
            .sort_values("rhm_Ntrain")
        )
        ax.plot(
            mean_df["rhm_Ntrain"] / pstar,
            mean_df["classifier_test_acc_at_best_val"],
            color=color,
            label=f"$m={int(m)}$",
        )

    ph.logx(ax)
    ax.set_xlabel(r"Rescaled number of samples, $P \,/\, " + pstar_latex + "$")
    if m_L_exp is not None:
        ax.set_xlabel(rf"Rescaled number of samples, $\,/\, m^{{{m_L_exp}}}$")
    ax.set_ylabel("Classification accuracy")

    if insets:
        ph.logx(axins)
        axins.minorticks_off()
        axins.set_xlabel("Num. Samples, $P$", fontsize=INSET_LABELSIZE)
        axins.set_yticklabels([])
        axins.tick_params(axis="both", which="both", labelsize=INSET_TICKSIZE, pad=1)

        axins.annotate(
            "",
            xy=(0.7, 0.15),
            xytext=(0.2, 0.5),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="black"),
        )
        axins.text(
            0.2,
            0.52,
            rf"$m={int(min(m_values))}$",
            transform=axins.transAxes,
            fontsize=INSET_ANNOTATION_SIZE,
            ha="center",
            va="bottom",
        )
        axins.text(
            0.7,
            0.2,
            rf"$m={int(max(m_values))}$",
            transform=axins.transAxes,
            fontsize=INSET_ANNOTATION_SIZE,
            ha="left",
            va="bottom",
        )

        axins2.set_xlabel(r"$m$", fontsize=INSET_LABELSIZE, labelpad=1)
        axins2.set_ylabel(r"$P^*$", fontsize=INSET_LABELSIZE, labelpad=1)
        axins2.tick_params(axis="both", which="both", labelsize=INSET_TICKSIZE, pad=1)
        axins2.set_yscale("log")
        axins2.set_xscale("log")
        axins2.minorticks_off()
        axins2.set_xticks([7, 14], [7, 14])

        ms = np.array(m_values)
        axins2.plot(ms, 0.6 * pstar_f(v, ms), "k--", zorder=-5)
        axins2.plot(ms, 1.2e-1 * np.array(ms) ** (L + 1), "r:", zorder=-5)
        axins2.annotate(
            r"$\propto m^{L+1}$",
            xy=(0.15, 0.75),
            xycoords="axes fraction",
            fontsize=INSET_ANNOTATION_SIZE,
            color="red",
        )
        axins2.annotate(
            r"$P^\ast \propto m^3$",
            xy=(0.95, 0.15),
            ha="right",
            xycoords="axes fraction",
            fontsize=INSET_ANNOTATION_SIZE,
        )
        axins2.set_ylim(2e3, 4e4)

def plot_e2e_statistician(ax, m_L_exp: int | None = None, insets: bool = True):
    if insets:
        axins = ax.inset_axes([0.04, 0.55, 0.40, 0.42])
        axins2 = ax.inset_axes([0.73, 0.15, 0.26, 0.35])
        df_subset = df_e2e_statistician.copy()
        normalized_df = pd.DataFrame()
        normalized_df["m"] = df_e2e_statistician["m"]
        normalized_df["P"] = df_subset["P"]
        normalized_df["acc"] = df_subset["accuracy"]
        normalized_df["seed"] = df_subset["seed"]
        normalized_df["chance_acc"] = 1 / df_subset["v"]
        plot_insets_pstar_sweep_and_fit(axins, axins2, normalized_df, diagnostics=False)
    m_values = sorted(df_e2e_statistician["m"].unique())
    L = df_e2e_statistician["L"].unique()[0]
    for m in m_values:
        color = get_color_sequential(m, m_values[::-1])
        m_df = df_e2e_statistician[df_e2e_statistician["m"] == m].sort_values(
            ["P", "seed"]
        )
        mean_df = (
            m_df.groupby("P", as_index=False)
            .agg(
                accuracy_mean=("accuracy", "mean"),
                P_over_vm3=("P_over_vm3", "first"),
                v=("v", "first"),
                m=("m", "first"),
            )
            .sort_values("P")
        )

        m_exp = m_L_exp if m_L_exp is not None else 3
        ax.plot(
            m_df["P"] / (m_df["v"] * m_df["m"] ** m_exp),
            m_df["accuracy"],
            "o",
            color=color,
            alpha=0.55,
            markersize=4,
            linestyle="none",
        )
        ax.plot(
            mean_df["P"] / (mean_df["v"] * mean_df["m"] ** m_exp),
            mean_df["accuracy_mean"],
            "-",
            color=color,
            label=rf"$m = {int(m)}$",
        )

    ax.set_xlabel(rf"Rescaled number of samples, $P/ {pstar_latex}$")
    if m_L_exp is not None:
        ax.set_xlabel(rf"Rescaled number of samples, $P/ m^{{{m_L_exp}}}$")
    ax.set_ylabel("Clustering accuracy")
    ph.logx(ax)
    if insets:
        ph.logx(axins)
        axins.minorticks_off()
        axins.set_xlabel(r"Num. Samples, $P$", fontsize=INSET_LABELSIZE, labelpad=1)
        axins.tick_params(axis="both", which="both", labelsize=INSET_TICKSIZE, pad=1)
        axins.set_xticks([1e2, 1e4, 1e6, 1e8])
        axins.set_yticklabels([])
        axins.tick_params(axis="y", which="both", left=False, right=False)
        axins.set_xlim(1e2, 1e8)
        axins.annotate(
            "",
            xy=(0.7, 0.15),
            xytext=(0.2, 0.5),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="black"),
        )
        axins.text(
            0.02,
            0.52,
            rf"$m={int(min(m_values))}$",
            transform=axins.transAxes,
            fontsize=INSET_ANNOTATION_SIZE,
            ha="left",
            va="bottom",
        )
        axins.text(
            0.7,
            0.2,
            rf"$m={int(max(m_values))}$",
            transform=axins.transAxes,
            fontsize=INSET_ANNOTATION_SIZE,
            ha="left",
            va="bottom",
        )

        axins2.set_xlabel(r"$m$", fontsize=INSET_LABELSIZE, labelpad=1)
        axins2.set_ylabel(r"$P^\ast$", fontsize=INSET_LABELSIZE, labelpad=1)
        axins2.tick_params(axis="both", which="both", labelsize=INSET_TICKSIZE, pad=1)
        axins2.set_yscale("log")
        axins2.set_xscale("log")
        axins2.minorticks_off()
        axins2.set_xticks([6, 12, 24], labels=[6, 12, 24])
        axins2.annotate(
            rf"$P^\ast \propto m^3$",
            xy=(10, 3e3),
            fontsize=INSET_ANNOTATION_SIZE,
        )
        axins2.annotate(
            r"$\propto m^{L+1}$",
            xy=(0.05, 0.85),
            fontsize=INSET_ANNOTATION_SIZE,
            xycoords="axes fraction",
            color="red",
        )

        ms = sorted(df_e2e_statistician["m"].unique())
        axins2.plot(
            ms,
            1.6 * pstar_f(df_e2e_statistician["v"].unique()[0], np.array(ms)),
            "k--",
            zorder=-5,
        )
        axins2.plot(
            ms,
            1.6e-2 * np.array(ms) ** (L + 1),
            color="red",
            ls=":",
            zorder=-5,
        )
        axins2.set_ylim(1e3, 1e6)

def main():
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    plot_e2e_statistician(axs[0])
    plot_ptc_m_sweep(axs[1])
    fig.tight_layout()
    savefig(fig, "clustering_and_ptc_m_sweeps")

if __name__ == "__main__":
    main()
