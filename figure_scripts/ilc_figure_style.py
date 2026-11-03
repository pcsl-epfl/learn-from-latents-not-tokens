"""Editable plotting/layout code for Figure 2; data come from fig2.py.

fig2.py owns sampling and clustering. This module only draws those results.
Times prose and Computer Modern mathematics match the NeurIPS manuscript.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from sklearn.manifold import TSNE


INK = "#202020"
BORDER = "#D1D1D1"
HEADER = "#F0F0F0"
CLUSTER = "#E6E6E6"
GREEK = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
    "nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
).split()

PLAYGROUND_LAYOUT = dict(figure_size=(5.5, 3.5))


def configure_style():
    plt.style.use("tableau-colorblind10")
    plt.rcParams.update({
        "font.family": "serif",
        # Selecting Times through Matplotlib would load mathptmx, which also
        # changes mathematics. The paper instead uses times.sty for prose only.
        "font.serif": ["Computer Modern Roman"],
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{times}",
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 6,
        "legend.fontsize": 7,
        "lines.linewidth": 1.2,
        "lines.markersize": 5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def symbol_names(v):
    return [chr(97 + i % 26) + (f"_{{{i // 26}}}" if i >= 26 else "") for i in range(v)]


def latent_names(v):
    return [
        ("o" if GREEK[i % 24] == "omicron" else "\\" + GREEK[i % 24])
        + (f"_{{{i // 24}}}" if i >= 24 else "")
        for i in range(v)
    ]


def panel_frame(ax, letter, title, *, header_height=0.18):
    """Figure 1's white panel, gray title strip, and serif heading."""
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=.025",
        facecolor="white", edgecolor=BORDER, linewidth=.7,
        transform=ax.transAxes, clip_on=False, zorder=-10,
    ))
    ax.add_patch(Rectangle(
        (.002, 1 - header_height), .996, header_height - .002,
        facecolor=HEADER, edgecolor="none", transform=ax.transAxes, zorder=-9,
    ))
    ax.text(.015, 1 - header_height / 2,
            rf"\textbf{{{letter}\quad {title}}}",
            transform=ax.transAxes, va="center", fontsize=8.5, color=INK)


def draw_algorithm(ax):
    """One lettered box contains the foreach and the complete ILC loop."""
    panel_frame(ax, "a", "ILC Algorithm", header_height=.19)
    ax.text(.018, .48, "For each level\n" + r"$\ell=0,\ldots,L-2$:",
            fontsize=7.0, va="center", linespacing=1.2, color=INK)
    boxes = [
        (.205, .205, r"Observe level $\ell$",
         "Form tuples $T$ from\ncurrent representations"),
        (.455, .245, r"Predictor $p$",
         r"Compute $P(Z\mid T)$ for" + "\nsame-level cousin targets"),
        (.750, .230, r"Clusterer $C$",
         r"Cluster $T$ by $P(Z\mid T)$;" + "\n" + r"labels give level $\ell+1$"),
    ]
    for x, width, heading, body in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, .29), width, .43, boxstyle="round,pad=.005,rounding_size=.016",
            facecolor="white", edgecolor="#888888", linewidth=.6,
        ))
        ax.text(x + width / 2, .605, heading, ha="center", va="center", fontsize=7.6)
        ax.text(x + width / 2, .425, body, ha="center", va="center", fontsize=6.5,
                linespacing=1.15, color="#444444")
    for start, end in [(.415, .450), (.705, .745)]:
        ax.add_patch(FancyArrowPatch(
            (start, .51), (end, .51), arrowstyle="->", mutation_scale=9,
            linewidth=1, color=INK,
        ))
    path = Path([(.865, .275), (.865, .115), (.23, .115), (.23, .275)],
                [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO])
    ax.add_patch(FancyArrowPatch(
        path=path, arrowstyle="-|>", mutation_scale=7,
        linewidth=.9, color="#555555", joinstyle="round",
    ))
    ax.text(.555, .115, r"Lift to level $\ell+1$ and repeat", ha="center", va="center",
            fontsize=7, bbox=dict(facecolor="white", edgecolor="none", pad=2))


def draw_context_tree(ax, stats, idx, name="", colors=None, show_labels=True):
    """A local cousin-prediction tree in Figure 1's visual grammar.

    T is the tuple being clustered, Z is the cousin prediction target, and X
    is T's parent. All three labels are shown on every row; colors are unused.
    """
    ax.set(xlim=(-.16, 1.15), ylim=(.17, 1.01))
    ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.set_axis_off()
    points = [(.50, .91), (.23, .62), (.77, .62),
              (.09, .34), (.37, .34), (.63, .34), (.91, .34)]
    # The same tuple-to-parent shaded grouping as Figure 1.
    cluster_path = Path(
        [(.69, .69), (.77, .74), (.85, .69),
         (1.00, .35), (1.04, .24), (.96, .24),
         (.58, .24), (.50, .24), (.54, .35), (.69, .69)],
        [Path.MOVETO, Path.CURVE3, Path.CURVE3,
         Path.LINETO, Path.CURVE3, Path.CURVE3,
         Path.LINETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY],
    )
    ax.add_patch(PathPatch(cluster_path, facecolor=CLUSTER, edgecolor="none", zorder=0))
    # A white outline groups the two symbols forming the prediction target Z.
    ax.add_patch(FancyBboxPatch(
        (.015, .255), .43, .17, boxstyle="round,pad=.01,rounding_size=.018",
        facecolor="white", edgecolor=INK, linewidth=.7, zorder=1,
    ))
    for parent, children in [(0, (1, 2)), (1, (3, 4)), (2, (5, 6))]:
        px, py = points[parent]
        rule_y = py - .105
        ax.plot([px, px], [py - .06, rule_y + .023], color=INK, lw=.8, zorder=2)
        for child in children:
            cx, cy = points[child]
            ax.plot([px, cx], [rule_y - .023, cy + .060], color=INK, lw=.8, zorder=2)
        ax.add_patch(Rectangle(
            (px - .023, rule_y - .023), .046, .046,
            facecolor="white", edgecolor=INK, linewidth=.7, zorder=3,
        ))
    symbols, latents = symbol_names(stats["v"]), latent_names(stats["v"])
    tokens = stats["tuples"][idx]
    for i, (x, y) in enumerate(points):
        ax.add_patch(Circle((x, y), .060, facecolor="white", edgecolor=INK,
                            linewidth=.8, zorder=4))
        label = latents[idx // stats["m"]] if i == 2 else (
            symbols[tokens[i - 5]] if i in (5, 6) else "")
        if label:
            ax.text(x, y, f"${label}$", ha="center", va="center", fontsize=7.0, zorder=5)
    if stats["L"] > 2:
        ax.plot([.50, .50], [.97, 1.01], color=INK, lw=.7, ls=":", clip_on=False)
    ax.text(-.105, .34, r"$Z$", ha="center", va="center", fontsize=8.0)
    ax.text(1.105, .34, r"$T$", ha="center", va="center", fontsize=8.0)
    ax.text(.925, .62, r"$X$", ha="left", va="center", fontsize=8.0)


def draw_probability_panel(ax, stats, idx, *, show_exact=True, **_):
    displayed = stats["displayed_bins"]
    positions = np.arange(len(displayed))
    symbols = symbol_names(stats["v"])
    parent = idx // stats["m"]
    if stats["incidence"][idx] > 0:
        ax.bar(positions, stats["empirical"][idx, displayed],
               color=stats["tuple_colors"][idx], width=1., edgecolor="none", zorder=2)
    if show_exact and len(displayed) and stats["parent_mass"][parent] > 0:
        ax.stairs(stats["exact"][parent, displayed], np.arange(len(displayed) + 1) - .5,
                  color=INK, linestyle="--", linewidth=1., fill=False, zorder=5)
    heights = [stats["empirical"][i, displayed].max()
               for i in stats["chosen"] if stats["incidence"][i] > 0]
    if show_exact:
        heights += [stats["exact"][i // stats["m"], displayed].max()
                    for i in stats["chosen"] if stats["parent_mass"][i // stats["m"]] > 0]
    ax.set(xlim=(-.6, max(1, len(displayed)) - .4),
           ylim=(0, max([float(x) for x in heights] + [.1]) * 1.5),
           ylabel=r"$P(Z\mid T)$")
    ax.set_xticks(positions, [rf"${symbols[a]}{symbols[b]}$"
                             for a, b in stats["target_tuples"][displayed]], fontsize=7)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0, labelsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BORDER)
    if stats["truncated"]:
        ax.text(1.02, -.07, r"$\ldots$", transform=ax.transAxes, va="top", fontsize=9)


def draw_shared_legend(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    label_y, key_y = .55, .07
    for x, text in [(.005, r"$T$: tuple to cluster"),
                    (.36, r"$Z$: prediction target"),
                    (.72, r"$X$: parent latent")]:
        ax.text(x, label_y, text, fontsize=6.6, va="center", color="#555555")
    ax.plot([.015], [key_y], marker="s", markersize=3.6, markerfacecolor="white",
            markeredgecolor=INK, markeredgewidth=.6)
    ax.text(.045, key_y, "production rule", fontsize=6.3, va="center", color="#555555")
    ax.add_patch(FancyBboxPatch((.34, key_y - .10), .035, .20,
                              boxstyle="round,pad=0,rounding_size=.02",
                              facecolor=CLUSTER, edgecolor="none", clip_on=False))
    ax.text(.392, key_y, "cluster into $X$", fontsize=6.3, va="center", color="#555555")
    ax.plot([.71, .76], [key_y, key_y], color=INK, ls="--", lw=.9)
    ax.text(.785, key_y, r"exact $P(Z\mid X)$", fontsize=6.3, va="center", color="#555555")


def draw_rhm_rule_table_panel(ax, rules, *, row_colors=None):
    L, m = len(rules), rules[0].shape[1]
    v = max(1 + int(rule.max()) for rule in rules.values())
    entries = []
    for ell in range(L):
        entries.append((rf"Parent level $\ell={L - ell}$", "", ell, None))
        parent_labels = [str(i) for i in range(rules[ell].shape[0])] if ell == 0 else latent_names(v)
        child_labels = symbol_names(v) if ell == L - 1 else latent_names(v)
        for parent in range(rules[ell].shape[0]):
            productions = []
            for synonym in range(m):
                rhs = r"\,".join(child_labels[int(token)] for token in rules[ell][parent, synonym])
                productions.append(rf"{parent_labels[parent]}\to {rhs}")
            entries.append((rf"${parent_labels[parent]}$",
                            "$" + r",\quad ".join(productions) + "$", ell, parent))
    table = ax.table(cellText=[row[:2] for row in entries], colWidths=[.09, .91],
                     cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(6.3)
    palette = plt.get_cmap("Pastel2")(np.linspace(0, 1, max(L, 2)))
    for row_id, (label, _, ell, parent) in enumerate(entries):
        is_level = parent is None
        face = "#EEEEEE" if is_level else "#F5F5F5" if ell == 0 else (
            row_colors[parent] if row_colors is not None else palette[ell])
        for col in range(2):
            cell = table[(row_id, col)]
            cell.set_facecolor(face)
            cell.set_edgecolor("white")
            if is_level:
                cell.get_text().set_text("")
        if is_level:
            ax.text(.5, 1 - (row_id + .5) / len(entries), label,
                    transform=ax.transAxes, ha="center", va="center", fontsize=6.3)
    ax.set_axis_off()
    return table


def draw_tsne_panel(ax, stats):
    valid = stats["incidence"] > 0
    ids = np.flatnonzero(valid)
    symbols, latents = symbol_names(stats["v"]), latent_names(stats["v"])
    if len(ids) >= 3:
        xy = TSNE(n_components=2, perplexity=min(8., (len(ids) - 1) / 3),
                  init="random", learning_rate=50., random_state=19).fit_transform(stats["empirical"][valid])
        ax.scatter(*xy.T, c=stats["tuple_colors"][ids], s=25, edgecolors="white", linewidths=.5)
        # Labels sit inside the plot margins and remain clear of each point.
        for idx, offset in zip(stats["chosen"], [(-8, 19), (-18, -9), (17, -12)]):
            location = np.flatnonzero(ids == idx)
            if len(location):
                rhs = r"\,".join(symbols[token] for token in stats["tuples"][idx])
                ax.annotate(rf"${latents[idx // stats['m']]}\to {rhs}$", xy[location[0]],
                            xytext=offset, textcoords="offset points", fontsize=6.3,
                            ha="center", va="center",
                            arrowprops=dict(arrowstyle="->", color=INK, lw=.65))
        ax.margins(x=.30, y=.30)
    else:
        ax.text(.5, .5, "Need at least three\nobserved tuples", transform=ax.transAxes, ha="center")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(.02, 1.02, "Coloured by k-means", transform=ax.transAxes,
            fontsize=6.3, color="#555555", va="bottom")


def make_layout_playground(stats, grammar_rules, *, layout=PLAYGROUND_LAYOUT, rule_colors=None):
    """Lettered algorithm, three histogram/tree rows, rules, and enlarged t-SNE."""
    fig = plt.figure(figsize=layout["figure_size"], facecolor="white")
    # Preserve readable point sizes while reducing physical height by 27%.
    algorithm_ax = fig.add_axes([.015, .74, .97, .245])
    draw_algorithm(algorithm_ax)
    histogram_axes, tree_axes = [], []
    symbols = symbol_names(stats["v"])
    for letter, bottom, idx in zip("bcd", [.555, .36, .165], stats["chosen"]):
        ax = fig.add_axes([.105, bottom, .405, .15])
        tree_ax = fig.add_axes([.525, bottom - .01, .21, .18])
        draw_probability_panel(ax, stats, idx)
        draw_context_tree(tree_ax, stats, idx)
        tuple_value = ",".join(symbols[token] for token in stats["tuples"][idx])
        ax.text(.015, .97, rf"\textbf{{{letter}.}}\quad $T=({tuple_value})$",
                transform=ax.transAxes, fontsize=8.2, va="top")
        histogram_axes.append(ax)
        tree_axes.append(tree_ax)
    histogram_axes[-1].set_xlabel(r"Prediction target tuple $Z$", fontsize=7, labelpad=1)
    # The key sits below the shared x-axis title while its bottom stays aligned
    # with t-SNE; the figure's physical height remains unchanged.
    legend_ax = fig.add_axes([.10, .018, .63, .10])
    draw_shared_legend(legend_ax)
    table_frame = fig.add_axes([.755, .42, .23, .30])
    panel_frame(table_frame, "e", "Production rules", header_height=.16)
    table_ax = table_frame.inset_axes([.01, .01, .98, .82])
    draw_rhm_rule_table_panel(table_ax, grammar_rules, row_colors=rule_colors)
    tsne_frame = fig.add_axes([.755, .025, .23, .37])
    panel_frame(tsne_frame, "f", "t-SNE", header_height=.12)
    tsne_ax = tsne_frame.inset_axes([.04, .025, .92, .77])
    draw_tsne_panel(tsne_ax, stats)
    return dict(figure=fig, algorithm_ax=algorithm_ax,
                histogram_axes=histogram_axes, tree_insets=tree_axes,
                table_ax=table_ax, tsne_ax=tsne_ax, legend_ax=legend_ax)
