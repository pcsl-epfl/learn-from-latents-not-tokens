"""Per-seed logistic fits and recomputed Monte Carlo uncertainty."""
from common import *
import pandas as pd
import scipy.optimize as opt

def plot_insets_pstar_sweep_and_fit(
    axins1, axins2, df, thresh=0.5, min_fit_m=0, diagnostics=True
):
    rng = np.random.default_rng(0)
    m_values = sorted(df["m"].unique())
    df["scaled_acc"] = (df["acc"] - df["chance_acc"]) / (1.0 - df["chance_acc"])
    logistic = lambda P, k, P0: 1 / (1 + np.exp(-k * (P - P0)))

    def get_curve_fit(group):
        popt, pcov = opt.curve_fit(
            logistic,
            np.log(group["P"]),
            group["scaled_acc"],
            p0=[1.0, np.median(np.log(group["P"]))],
            maxfev=10000,
        )
        k, P0 = popt
        return pd.Series({"k": k, "P0": P0})

    fitted_df = df.groupby(["m", "seed"], as_index=False).apply(
        get_curve_fit, include_groups=False
    )

    fitted_df_means = fitted_df.groupby("m", as_index=False).apply(
        lambda group: pd.Series(
            {
                "logP0_mean": np.mean(group["P0"]),
                "logP0_std": np.std(group["P0"], ddof=1),
                "Nseeds": len(group),
            }
        ),
        include_groups=False,
    )
    fitted_df_means = fitted_df_means[fitted_df_means["m"] >= min_fit_m]

    def get_linear_fit_slope(logP0, m):
        coeffs = np.polyfit(
            np.log(m),
            logP0,
            1,
        )
        return coeffs

    Ntrials = 300
    sampled_P0s = [
        rng.normal(
            loc=logP0_mean, scale=logP0_std, size=Ntrials * int(Nseeds)
        ).reshape(Ntrials, int(Nseeds))
        for m, logP0_mean, logP0_std, Nseeds in zip(
            fitted_df_means["m"],
            fitted_df_means["logP0_mean"],
            fitted_df_means["logP0_std"],
            fitted_df_means["Nseeds"],
        )
    ]
    slopes = []
    intercepts = []
    for trial in range(Ntrials):
        trial_logP0s = [sampled_P0[trial] for sampled_P0 in sampled_P0s]
        trial_ms = [
            np.ones_like(sampled_P0[trial]) * m
            for sampled_P0, m in zip(sampled_P0s, fitted_df_means["m"])
        ]
        slope, intercept = get_linear_fit_slope(
            np.array(trial_logP0s).flatten(),
            np.array(trial_ms).flatten(),
        )
        slopes.append(slope)
        intercepts.append(intercept)
    print(
        f"Estimated slope and uncertainty uncertainty from MC: {np.mean(slopes):.3f}, {np.std(slopes, ddof=1):.3f}"
    )
    print(
        f"Upper and lower bounds on slope: {np.percentile(slopes, 2.5):.3f}, {np.percentile(slopes, 97.5):.3f}"
    )
    print(
        f"Upper and lower error on slope: {np.percentile(slopes, 2.5) - np.mean(slopes):.3f}, {np.percentile(slopes, 97.5) - np.mean(slopes):.3f}"
    )
    print(
        f"Estimated intercept and uncertainty from MC: {np.mean(intercepts):.3f}, {np.std(intercepts, ddof=1):.3f}"
    )

    for m in m_values:
        df_m = df[df["m"] == m]

        mean_df = df_m.groupby("P", as_index=False)["acc"].mean().sort_values("P")

        color = get_color_sequential(m, m_values[::-1])
        axins1.scatter(
            df_m["P"],
            df_m["acc"],
            alpha=0.55,
            color=color,
            edgecolor="none",
            label=None,
        )
        fitted_m_df = fitted_df[fitted_df["m"] == m]
        for seed in fitted_m_df["seed"].unique():
            fitted_seed_df = fitted_m_df[fitted_m_df["seed"] == seed]
            k, P0 = fitted_seed_df.iloc[0]["k"], fitted_seed_df.iloc[0]["P0"]
            P_fit = np.geomspace(df_m["P"].min(), df_m["P"].max(), 100)
            acc_fit = (
                logistic(np.log(P_fit), k, P0) * (1.0 - df_m["chance_acc"].iloc[0])
                + df_m["chance_acc"].iloc[0]
            )
            if diagnostics:
                axins1.plot(P_fit, acc_fit, color=color, alpha=0.3, ls="--")

        axins1.plot(
            mean_df["P"],
            mean_df["acc"],
            color=color,
            label=f"$m={int(m)}$",
            alpha=1.0,
        )
        if diagnostics:
            acc = mean_df["acc"].values
            ps = mean_df["P"].values
            sort_idx = np.argsort(acc)
            p_star_val = np.interp(thresh, acc[sort_idx], ps[sort_idx])
            axins2.plot(m, p_star_val, "x", color=color, markersize=4)

        mu = fitted_m_df["P0"].mean()
        sigma = fitted_m_df["P0"].std(ddof=1)
        yerrs = np.array(
            [[np.exp(mu + sigma) - np.exp(mu), np.exp(mu) - np.exp(mu - sigma)]]
        ).T
        axins2.errorbar(
            m,
            np.exp(fitted_m_df["P0"].mean()),
            yerr=yerrs,
            fmt="o",
            color=color,
            markersize=4,
            capsize=2,
        )
    axins2.plot(
        fitted_df_means["m"],
        fitted_df_means["m"] ** np.mean(slopes) * np.exp(np.mean(intercepts)),
        color="grey",
        ls=":",
        lw=1.5,
    )
    ph.loglog(axins1)
    ph.loglog(axins2)
    return fitted_df_means
