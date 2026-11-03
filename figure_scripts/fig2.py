"""Main Figure 2: fixed-seed sampling and clustering illustration."""
from common import *
import random
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import illustration_rhm as rhm
from ilc_figure_style import configure_style, draw_algorithm, draw_context_tree, make_layout_playground, PLAYGROUND_LAYOUT
S = 2
configure_style()
plt.rcParams["text.usetex"] = os.environ.get("USE_TEX", "1") != "0"
FULL_WIDTH = 8
FIGURE_HEIGHT = 4.0
INSET_LABELSIZE = 8
INSET_TICKSIZE = 7
INSET_ANNOTATION_SIZE = 8
HISTOGRAM_XTICK_LABELSIZE = 8
GREEK = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega").split()
def symbol_names(v):
    return [chr(97 + i % 26) + (f"_{{{i // 26}}}" if i >= 26 else "") for i in range(v)]


def latent_names(v):
    return [
        ("o" if GREEK[i % 24] == "omicron" else "\\" + GREEK[i % 24])
        + (f"_{{{i // 24}}}" if i >= 24 else "")
        for i in range(v)
    ]


def latent_options(v):
    # Native HTML dropdown options cannot typeset math; use readable names here.
    greek = "αβγδεζηθικλμνξοπρστυφχψω"
    return [(greek[i % 24] + (f"_{i // 24}" if i >= 24 else ""), i) for i in range(v)]


def plot_palette(v, m):
    if v <= 12:
        bases = plt.get_cmap("Pastel2")(np.arange(v))[:, :3]
    else:
        from matplotlib.colors import hsv_to_rgb

        bases = hsv_to_rgb(
            np.column_stack(
                [(np.arange(v) * 0.61803398875) % 1, np.full(v, 0.38), np.full(v, 0.87)]
            )
        )
    colors = np.array(
        [
            np.clip(base * (0.82 + 0.18 * j / (m - 1)) + 0.08 * j / (m - 1), 0, 1)
            for base in bases
            for j in range(m)
        ]
    )
    return bases, colors

def context_statistics(
    rule_seed=42, num_samples=100_000, sampling_seed=7, v=12, m=4, L=2
):
    """One fixed cousin pair per sample; IDs are parent*M + synonym."""
    if any(not isinstance(x, (int, np.integer)) for x in (v, m, L, num_samples)):
        raise ValueError("v, m, L, and num_samples must be integers")
    if v < 2 or not 2 <= m <= v or L < 2:
        raise ValueError(
            "Require v >= 2, 2 <= m <= v (two distinct synonyms), and L >= 2."
        )
    V, M = v, m
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    # rhm.sample_rules seeds Python's global RNG; restore its state afterwards.
    state = random.getstate()
    try:
        rules = rhm.sample_rules(V, V, M, S, L=L, seed=int(rule_seed))
    finally:
        random.setstate(state)
    top, bottom = (rules[i].numpy() for i in (L - 2, L - 1))
    tuples = bottom.reshape(V * M, S)

    # Independent RNG streams give nested sample prefixes as P increases.
    streams = [
        np.random.default_rng(s)
        for s in np.random.SeedSequence(int(sampling_seed)).spawn(L + 2)
    ]
    root = streams[0].integers(V, size=num_samples)
    root_rule = streams[1].integers(M, size=num_samples)
    # Follow child 0 until reaching the grandparent of leaves x0,...,x3.
    grandparent = root
    grandparent_mass = np.full(V, 1 / V)
    for depth in range(L - 2):
        layer = rules[depth].numpy()
        choice = streams[4 + depth].integers(M, size=num_samples)
        grandparent = layer[grandparent, choice, 0]
        grandparent_mass = np.bincount(
            layer[..., 0].ravel(),
            weights=np.repeat(grandparent_mass / M, M),
            minlength=V,
        )
    parents = top[grandparent, root_rule]
    left_rule = streams[2].integers(M, size=num_samples)
    right_rule = streams[3].integers(M, size=num_samples)
    observed_z = bottom[parents[:, 0], left_rule]
    # Decode the observed tuple into a unique rule ID (not a parent class).
    lookup = np.full((V, V), -1, dtype=int)
    lookup[tuple(tuples.T)] = np.arange(V * M)
    observed_t = bottom[parents[:, 1], right_rule]
    ids = lookup[tuple(observed_t.T)]
    # Columns are actual visible pairs, ordered lexicographically. Store only
    # the vm grammatical pairs; the remaining v**2-vm columns are identically zero.
    target_tuples = tuples[np.lexsort((tuples[:, 1], tuples[:, 0]))]
    target_lookup = np.full((V, V), -1, dtype=int)
    target_lookup[tuple(target_tuples.T)] = np.arange(V * M)
    z_ids = target_lookup[tuple(observed_z.T)]
    counts = np.bincount(ids * (V * M) + z_ids, minlength=(V * M) ** 2).reshape(
        V * M, V * M
    )
    incidence = counts.sum(axis=1)
    empirical = np.full(counts.shape, np.nan, dtype=float)
    np.divide(counts, incidence[:, None], out=empirical, where=incidence[:, None] > 0)

    # joint[b,a] = P(left parent=b, right parent=a).
    joint = np.zeros((V, V))
    np.add.at(
        joint,
        (top[..., 0].ravel(), top[..., 1].ravel()),
        np.repeat(grandparent_mass / M, M),
    )
    # P(Z=(z1,z2) | left parent=b): each of its m rules has mass 1/m.
    channel = np.zeros((V, V * M))
    channel[np.arange(V)[:, None], target_lookup[bottom[..., 0], bottom[..., 1]]] = (
        1 / M
    )
    parent_mass = joint.sum(axis=0)
    exact = np.full((V, V * M), np.nan)
    np.divide(
        joint.T @ channel,
        parent_mass[:, None],
        out=exact,
        where=parent_mass[:, None] > 0,
    )
    return dict(
        tuples=tuples,
        target_tuples=target_tuples,
        counts=counts,
        incidence=incidence,
        empirical=empirical,
        exact=exact,
        parent_mass=parent_mass,
        rules=rules,
        rule_seed=rule_seed,
        sampling_seed=sampling_seed,
        num_samples=num_samples,
        v=v,
        m=m,
        L=L,
    )


def illustrate(
    rule_seed=42,
    num_samples=100_000,
    sampling_seed=7,
    parent=0,
    synonym=0,
    other_parent=1,
    show_exact=True,
    v=12,
    m=4,
    L=2,
    max_bins=15,
    figure_height=FIGURE_HEIGHT,
):
    if (
        not isinstance(figure_height, (int, float, np.number))
        or not np.isfinite(figure_height)
        or figure_height <= 0
    ):
        raise ValueError("figure_height must be a positive finite number of inches.")
    if max_bins is not None and (
        not isinstance(max_bins, (int, np.integer)) or max_bins < 1
    ):
        raise ValueError("max_bins must be a positive integer or None (all bins).")
    if parent == other_parent:
        raise ValueError("Choose a different parent for the third histogram.")
    stats = context_statistics(rule_seed, num_samples, sampling_seed, v=v, m=m, L=L)
    V, M = v, m
    if not (0 <= parent < V and 0 <= other_parent < V and 0 <= synonym < M):
        raise ValueError(
            "Parent and synonym selections must be inside the selected vocabulary/rule range."
        )
    SYMBOLS = symbol_names(V)
    LATENTS = latent_names(V)
    BASE_COLORS, COLORS = plot_palette(V, M)
    empirical, incidence = stats["empirical"], stats["incidence"]
    chosen = [parent * M + synonym, parent * M + (synonym + 1) % M, other_parent * M]
    # The display filter is shared across all three rows and is empirical only.
    # Keep full context vectors for clustering, independent of example selection.
    occupied_bins = np.flatnonzero(stats["counts"][chosen].sum(axis=0) > 0)
    displayed_bins = occupied_bins if max_bins is None else occupied_bins[:max_bins]
    truncated = len(displayed_bins) < len(occupied_bins)
    tick_labels = [
        rf"${{{SYMBOLS[a]}}}{{{SYMBOLS[b]}}}$"
        for a, b in stats["target_tuples"][displayed_bins]
    ]
    positions = np.arange(len(displayed_bins))
    valid = incidence > 0
    ids = np.flatnonzero(valid)
    # Assign colors once from k-means, before drawing any panel.
    # Every occurrence of a tuple then uses its cluster color.
    ari = np.nan
    cluster_labels = ids // M
    if len(ids) >= V:
        raw_cluster_labels = KMeans(
            n_clusters=V, n_init=20, random_state=19
        ).fit_predict(empirical[valid])
        true_parents = ids // M
        ari = adjusted_rand_score(true_parents, raw_cluster_labels)
        # K-means labels are arbitrary. Match them to Greek parent labels by
        # the maximum-overlap permutation so legend and dropdown agree.
        overlap = np.zeros((V, V), dtype=int)
        np.add.at(overlap, (raw_cluster_labels, true_parents), 1)
        cluster_ids, parent_ids = linear_sum_assignment(-overlap)
        cluster_to_parent = np.arange(V)
        cluster_to_parent[cluster_ids] = parent_ids
        cluster_labels = cluster_to_parent[raw_cluster_labels]
    TUPLE_COLORS = COLORS.copy()
    TUPLE_COLORS[ids] = BASE_COLORS[cluster_labels % V]
    fig = plt.figure(
        figsize=(FULL_WIDTH, figure_height), layout="constrained", facecolor="white"
    )
    outer = fig.add_gridspec(2, 1, height_ratios=[1.45, 8.2], hspace=0.035)
    algorithm_ax = fig.add_subplot(outer[0])
    draw_algorithm(algorithm_ax)
    gs = outer[1].subgridspec(3, 3, width_ratios=[0.64, 1.66, 0.90], wspace=0.02)
    axes = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    embedding_ax = fig.add_subplot(gs[:, 2])
    tree_axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    names = ["T", r"T^{\prime}", r"T^{\prime\prime}"]
    for row, (tree_ax, idx, name) in enumerate(zip(tree_axes, chosen, names)):
        draw_context_tree(
            tree_ax, stats, idx, name, TUPLE_COLORS, show_labels=(row == 0)
        )
    heights = [empirical[i] for i in chosen if valid[i]]
    if show_exact:
        heights += [
            stats["exact"][i // M] for i in chosen if stats["parent_mass"][i // M] > 0
        ]
    ymax = max([float(h.max()) for h in heights] + [0.1]) * 1.45
    names = ["T", r"T^{\prime}", r"T^{\prime\prime}"]
    for ax, idx, name in zip(axes, chosen, names):
        a = idx // M
        if valid[idx]:
            ax.bar(
                positions,
                empirical[idx, displayed_bins],
                color=TUPLE_COLORS[idx],
                width=1.0,
                edgecolor="none",
                zorder=2,
            )
        else:
            ax.text(0.5, 0.5, "Tuple not observed", transform=ax.transAxes, ha="center")
        if show_exact and len(displayed_bins) and stats["parent_mass"][a] > 0:
            ax.stairs(
                stats["exact"][a, displayed_bins],
                np.arange(len(displayed_bins) + 1) - 0.5,
                color="#252525",
                linestyle="--",
                linewidth=1.3,
                fill=False,
                zorder=5,
            )
        label = ", ".join(SYMBOLS[t] for t in stats["tuples"][idx])
        rule = (
            rf"${LATENTS[a]} \to "
            + r"\,".join(SYMBOLS[t] for t in stats["tuples"][idx])
            + "$"
        )
        ax.text(
            0.025,
            0.94,
            rule,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=INSET_LABELSIZE,
            zorder=8,
            bbox=dict(
                boxstyle="round,pad=.3",
                facecolor=tuple(0.15 * TUPLE_COLORS[idx] + 0.85),
                edgecolor="none",
            ),
        )
        ax.set(
            ylim=(0, ymax),
            xlim=(-0.6, max(1, len(displayed_bins)) - 0.4),
            # ylabel=rf"$\phi(Z \mid T= {label})$",
            ylabel=rf"$P(Z \mid T= {label})$",
        )
        ax.set_xticks(
            positions,
            tick_labels,
            fontsize=HISTOGRAM_XTICK_LABELSIZE,
            rotation=90 if len(displayed_bins) > 24 else 0,
        )
        ax.tick_params(axis="x", length=0, labelsize=HISTOGRAM_XTICK_LABELSIZE)
        ax.tick_params(axis="y", length=0, labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#dddddd")
        if truncated:
            ax.annotate(
                "",
                xy=(1.035, 0),
                xytext=(0.98, 0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.0),
                annotation_clip=False,
            )
            ax.text(
                1.035,
                -0.065,
                r"$\ldots$",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=13,
                color="#777777",
                clip_on=False,
            )
    axes[-1].set_xlabel(r"Target tuple $Z$", fontsize=8)
    if show_exact:
        axes[0].legend(
            [Line2D([], [], color="#252525", ls="--")],
            [r"Exact $P(Z \mid X = h_\ell(T))$"],
            frameon=False,
            fontsize=8,
            loc="upper right",
        )
    if len(ids) >= 3:
        vectors = empirical[valid]
        perplexity = min(8.0, (len(ids) - 1) / 3)
        xy = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="random",
            learning_rate=50.0,
            random_state=19,
        ).fit_transform(vectors)
        point_colors = TUPLE_COLORS[ids]
        embedding_ax.scatter(
            *xy.T, c=point_colors, s=68, edgecolors="white", linewidths=0.65
        )
        annotation_offsets = [(-34, +6), (12, 18), (12, -24)]
        for idx, offset in zip(chosen, annotation_offsets):
            pos = np.flatnonzero(ids == idx)
            if len(pos):
                point = xy[pos[0]]
                parent_id = idx // M
                rule_tokens = stats["tuples"][idx]
                rule_label = (
                    rf"${LATENTS[parent_id]} \to "
                    + r"\,".join(SYMBOLS[token] for token in rule_tokens)
                    + "$"
                )
                embedding_ax.annotate(
                    rule_label,
                    point,
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=INSET_LABELSIZE,
                    ha="center",
                    va="center",
                    arrowprops={
                        "arrowstyle": "->",
                        # color=point_colors[pos[0]],
                        "color": "#252525",
                        "lw": 0.9,
                        "shrinkA": 2,
                        "shrinkB": 3,
                    },
                )
    else:
        embedding_ax.text(
            0.5,
            0.5,
            "Need at least 3 observed tuples for t-SNE",
            transform=embedding_ax.transAxes,
            ha="center",
            wrap=True,
        )
    embedding_ax.set_xticks([])
    embedding_ax.set_yticks([])
    embedding_ax.tick_params(length=0)
    for spine in embedding_ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#b5b5b5")
        spine.set_linewidth(0.8)
    embedding_ax.text(
        0.03,
        0.97,
        "tSNE of context vectors,\ncoloured by k-means",
        transform=embedding_ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        # bbox=dict(boxstyle="round,pad=.25", facecolor="white", edgecolor="none"),
        zorder=10,
    )
    stats.update(
        figure=fig,
        ari=ari,
        displayed_bins=displayed_bins,
        chosen=chosen,
        occupied_bins=occupied_bins,
        truncated=truncated,
        max_bins=max_bins,
        histogram_axes=axes,
        tree_axes=tree_axes,
        embedding_ax=embedding_ax,
        algorithm_ax=algorithm_ax,
        cluster_labels=cluster_labels,
        cluster_to_parent=(cluster_to_parent if len(ids) >= V else np.arange(V)),
        tuple_colors=TUPLE_COLORS,
        base_colors=BASE_COLORS,
        figure_height=figure_height,
    )
    return stats


def main():
    latest = illustrate(v=3, m=2, L=2, rule_seed=46, num_samples=500,
                        sampling_seed=7, parent=0, synonym=0, other_parent=2,
                        max_bins=15, figure_height=4.0, show_exact=True)
    plt.close(latest["figure"])
    layout = make_layout_playground(latest, latest["rules"], layout=PLAYGROUND_LAYOUT,
                                    rule_colors=latest["base_colors"])
    savefig(layout["figure"], "ilc_illustration")

if __name__ == "__main__":
    main()
