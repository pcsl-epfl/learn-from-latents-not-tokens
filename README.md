# Learn from your own latents and not from tokens

Reproduction code for the main-text figures and their simulation summaries.
The repository contains figure scripts, saved per-run/seed measurements, a minimal
gradient-based ILC implementation, and a historical iterative-clustering simulator.
Data2vec figures can be reproduced from saved summaries; data2vec training and
raw-log aggregation are deferred.

## Layout

- `figure_scripts/fig1.py`: editable vector schematic (Figure 1).
- `figure_scripts/fig2.py`: fixed-seed ILC illustration (Figure 2).
- `figure_scripts/fig3.py`: iterative and gradient-based ILC scaling (Figure 3).
- `figure_scripts/fig4.py`: data2vec sample-complexity panel (Figure 4).
- `figure_scripts/fig5.py`: data2vec synonym-clustering scores (Figure 5).
- `figure_scripts/tikz/`: data2vec schematic standalone LaTeX wrapper, and a precompiled PDF.
- `figure_scripts/make_all_figs.ipynb`: run all main-text figure scripts.
- `data/reproduction_data/`: versioned figure inputs and provenance manifest.
- `simulation/`: training, classifier evaluation, and aggregation scripts.
- `slurm/`: independent GPU and CPU array launchers (any reproduction effort will need to specialize this to available compute resources).

Generated figures, raw simulations, checkpoints, logs, and executed notebooks are
ignored by Git. Simulations write below `data/`; only `data/reproduction_data/`
is versioned. Appendix figures are outside this release's current scope.

## Dependencies

Python 3.12 was used for local validation. Python dependencies, in plain text:

numpy
pandas >= 2.2, < 3
pyarrow
matplotlib
scipy
scikit-learn
torch
einops
torchjd >= 0.7, < 0.9
reportlab
jupyter
nbconvert
ipykernel

TorchJD 0.7.0 was smoke-tested locally; the upstream Kuma environment has 0.8.1.
The code uses the public `torchjd.backward` API and UPGrad aggregation.
A CUDA-enabled PyTorch environment is needed for full gradient-based sweeps.
No package installation or environment activation occurs inside batch jobs.

The figure workflow also requires a TeX installation providing `latexmk`,
`pdflatex`, TikZ, `standalone`, Times and Computer Modern fonts. Matplotlib's
LaTeX rendering may require `dvipng`. Figure 1 uses Times New Roman when present
in `/System/Library/Fonts/Supplemental`; otherwise it falls back to PDF's built-in
Times fonts. Use `--font-dir PATH` with `fig1.py` to select your own Times New
Roman installation. The small bundled mathematical fonts have their original
license in `figure_scripts/fonts/LICENSE_BAKOMA.txt`.

## Reproduce figures from saved summaries

From the repository root, in an environment with the dependencies above:

```bash
python figure_scripts/make_all.py
```

Outputs are standalone PDFs and preview PNGs in `figures/` (Figure 1 also exports
editable SVG). Figure 4 has two manuscript assets: the data2vec TikZ schematic
and `d2vec_main_online_with_pstar.pdf`. The TikZ schematic is compiled into
`figures/schematic_ssl_data2vec.pdf`; its build output is redirected to a logfile.
Existing manuscript files are never overwritten.

Alternatively execute the notebook, keeping outputs out of the source notebook:

```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=600 \
  --output make_all_figs.executed.ipynb figure_scripts/make_all_figs.ipynb
```

`FIGURE_OUTPUT=/absolute/path` overrides the output directory.
`REPRODUCTION_DATA=/absolute/path` selects a complete alternative input directory.
`USE_TEX=0` disables LaTeX rendering in the statistical plots for quick previews;
the publication illustration and standalone TikZ workflow still require TeX.

The saved files contain measurements before bootstrapping/Monte Carlo fitting.
Per-seed logistic fits and 300 Monte Carlo trials are recomputed by `fit_utils.py`
with a fixed RNG seed. Fitted slope diagnostics are printed during plotting;
bootstrap draws are not shipped.

## Generate simulation data

### Local smoke checks

These run small CPU experiments rather than the full paper sweeps:

```bash
SMOKE=1 DEVICE=cpu PYTHON=/absolute/path/to/python bash slurm/gradient_ilc.sh
SMOKE=1 PYTHON=/absolute/path/to/python bash slurm/iterative_ilc.sh
python simulation/aggregate_gradient.py --results data/gradient_ilc/smoke \
  --output data/smoke_summary/df_ptc_m_sweep_best.parquet --allow-incomplete
python simulation/aggregate_iterative.py --results data/iterative_ilc/smoke \
  --output data/smoke_summary/df_e2e_statistician.parquet --allow-incomplete
```

Smoke outputs live in separate ignored directories and must not replace the
published summaries. Runs refuse to overwrite existing observation/result files;
choose another `--output-root` (gradient) or `--output-dir` (iterative) for repeats.

### Slurm arrays

Submit from the repository root; create the log directory before submission:

```bash
mkdir -p data/logs
PYTHON=/absolute/path/to/environment/bin/python sbatch slurm/gradient_ilc.sh
PYTHON=/absolute/path/to/environment/bin/python sbatch slurm/iterative_ilc.sh
```

The interpreter must point to an existing environment containing the dependencies.
The scripts never source `.bashrc`, use conda activation, or depend on a scratch
variable. Resources and array limits are declared in `#SBATCH` lines. Edit these
for your cluster. The GPU launcher targets Kuma's `h100` partition; the CPU
launcher targets Jed's `bigmem` partition. Full arrays have **165 GPU tasks** and
**18 CPU tasks**; do not launch them for a smoke test. These full resource limits
are starting settings, not a guarantee that every large iterative task finishes
within the specified memory/time budget.

For a one-task cluster smoke test, override the array and resources as appropriate:

```bash
SMOKE=1 PYTHON=/absolute/path/to/python sbatch --array=1-1 \
  --time=00:10:00 slurm/gradient_ilc.sh
```

For a running array, submit from an immutable checkout pinned to the desired Git
commit. Each task prints the revision. Keep its code unchanged until all tasks
finish. `REPO_ROOT` can explicitly select that checkout.

The gradient sweep preserves the original `sweep_params.txt` as
`simulation/gradient_sweep.csv`: v=10, s=3, L=4; m in {7,8,10,12,15}; 11 sample
counts; three grammar seeds. A task trains the representation, then evaluates
all saved checkpoints with the original root classifier. Aggregation chooses the
checkpoint with greatest validation accuracy, breaking ties by latest step, and
reports its test accuracy. Historical `--wd` is retained, including upstream
Adam's behavior of not applying that argument.

The iterative sweep uses v=8, s=3, L=5, m in {6,8,12,16,22,32}, three grammar
seeds, and the historical 16-point sample grid. It retains the historical
algorithm and sampling functions, without belief-propagation code. K-means now
has a fixed seed (historical runs did not); fresh curves need not match old
measurements bit for bit. See `PROVENANCE.md` for source attribution and limits.

### Aggregate results

```bash
python simulation/aggregate_gradient.py
python simulation/aggregate_iterative.py
```

Outputs go to ignored `data/aggregated/`. Each aggregator writes a coverage report
and refuses incomplete results unless `--allow-incomplete` is specified. Failed
classifier checkpoints are reported rather than silently selected. The historical
iterative data lacks the largest sample point for m=32, for all three seeds.

To plot fresh non-data2vec results alongside the saved data2vec summaries:

```bash
cp data/reproduction_data/data2vec*.pkl data/aggregated/
REPRODUCTION_DATA="$PWD/data/aggregated" python figure_scripts/fig3.py
```

`simulation/export_saved_summaries.py --source PATH` exports the four main-text
inputs from an existing `make_summary_data.ipynb` output directory. It filters
out unused experimental conditions and retains no model tensors or bootstrap
replicates. Raw-log aggregation for data2vec is not included in this release.


### One Figure 3 experiment

`slurm/figure3_single.sh` runs L=4, s=3, v=10, m=8, P=30,000, grammar seed
1337 on one MIG GPU. The separate `simulation/figure3_single.csv` specifies
40 training/classifier epochs. Training has a graceful 20-minute cutoff inside
a 30-minute allocation; saved checkpoints are evaluated before aggregation.
This leaves the original full-sweep parameter file unchanged.

```bash
mkdir -p data/logs
PYTHON=/absolute/path/to/python sbatch slurm/figure3_single.sh
```

Outputs and the automatically generated summary/coverage report go to
`data/figure3_single_JOBID/`. To rerun aggregation alone:

```bash
python simulation/aggregate_gradient.py \
  --results data/figure3_single_JOBID/results \
  --expected-sweep simulation/figure3_single.csv \
  --output data/figure3_single_JOBID/summary/df_ptc_m_sweep_best.parquet
```

The runner also accepts `--sweep PATH`, `--max-training-seconds SECONDS`, and
`--checkpoint-min-step N` for other bounded experiments. A cutoff saves the
current model and observations before classifier evaluation; it does not imply
that convergence was reached.
