"""Historical iterative-clustering algorithm, extracted from statistician_e2e.py.
Sampling/algorithm preserved; paths configurable, smoke mode and fixed k-means RNG added.
The historical script did not fix the k-means RNG; saved curves are not bitwise reproducible.
"""
import torch
import argparse
import torch.nn.functional as F
import numpy as np
from sampling import sample_rules, torch_datum_allclasses
from pathlib import Path
import json
import os
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment


def compute_accuracy(labels, true_labels):

    labels = labels.numpy()
    true_labels = true_labels.numpy()

    cm = confusion_matrix(true_labels, labels)

    def _make_cost_m(cm):
        s = np.max(cm)
        return -cm + s

    indices = linear_assignment(_make_cost_m(cm))
    indices = np.asarray(indices)
    indices = np.transpose(indices)
    js = [e[1] for e in sorted(indices, key=lambda x: x[0])]
    cm2 = cm[:, js]
    best_accuracy = np.trace(cm2) / np.sum(cm2)

    return best_accuracy


parser = argparse.ArgumentParser()
parser.add_argument("--v", type=int, default=32)
parser.add_argument("--m", type=int, default=8)
parser.add_argument("--s", type=int, default=3)
parser.add_argument("--L", type=int, default=4)
parser.add_argument("--rules_seed", type=int, default=0)
parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data/iterative_ilc/results")
parser.add_argument("--smoke", action="store_true", default=os.environ.get("SMOKE", "0") == "1")
parser.add_argument("--kmeans-seed", type=int, default=0)
args = parser.parse_args()
if args.smoke:
    args.v, args.m, args.s, args.L = 4, 2, 3, 3
args.output_dir.mkdir(parents=True, exist_ok=True)
output_file = args.output_dir / f"statistician_e2e_v{args.v}_m{args.m}_s{args.s}_L{args.L}_rs{args.rules_seed}.pt"
if output_file.exists():
    raise FileExistsError(f"Refusing to overwrite {output_file}")
(output_file.with_suffix(".metadata.json")).write_text(json.dumps({**vars(args), "output_dir": str(args.output_dir)}, indent=2) + "\n")

print(args, flush=True)

v = args.v
n = v
m = args.m
s = args.s
L = args.L

rules = sample_rules(v, n, m, s, L, seed=args.rules_seed)

start = 512
end = min(v * (m**3) * 300, n * m ** ((s**L - 1) / (s - 1)))

log_spaced_points = torch.logspace(
    torch.log10(torch.tensor(start)), torch.log10(torch.tensor(end)), steps=16
).int()
log_spaced_points = torch.unique(log_spaced_points)
if args.smoke:
    log_spaced_points = torch.tensor([64, 128])

results = {}

for num_points in log_spaced_points:

    print(f"num_points: {num_points}", flush=True)

    x = torch_datum_allclasses(seed=0, rules=rules, v=v, m=m, ptr=num_points)

    xl = x[L]
    for l in range(L - 1):
        features = xl[:, :s]
        values = xl[:, s : s**2]
        single_featuresl = torch.unique(xl[:, :s], dim=0)
        means = []
        for feature in single_featuresl:
            mask = (features == feature).all(dim=1)
            filtered_values = values[mask]
            if filtered_values.size(0) == 0:
                means.append(torch.zeros((v * values.size(1),), dtype=torch.float))
            else:
                one_hot_encoded = F.one_hot(
                    filtered_values.long(), num_classes=v
                ).float()
                mean = one_hot_encoded.flatten(start_dim=1).mean(dim=0)
                means.append(mean)
        means = torch.stack(means)

        kmeans = KMeans(n_clusters=min(v,len(means)), n_init=100, random_state=args.kmeans_seed)
        kmeans.fit(means)
        labels = torch.tensor(kmeans.labels_).int()

        newxl = torch.zeros((xl.size(0), xl.size(1) // s), dtype=torch.int)
        xl = xl.reshape(-1, s)
        newxl = newxl.reshape(-1, 1)
        for id, feature in enumerate(single_featuresl):
            mask = (xl == feature).all(dim=1)
            newxl[mask] = labels[id]
        newxl = newxl.reshape(num_points.item(), -1)
        xl = newxl

    accuracy = compute_accuracy(xl.flatten(), x[1].flatten())
    print(f"accuracy: {accuracy}", flush=True)

    results[num_points.item()] = accuracy

    torch.save(
        results,
        output_file,
    )
