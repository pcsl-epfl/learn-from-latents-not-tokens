"""RHM rule sampler used by the fixed illustration."""
import random
from itertools import product
import torch

def sample_rules(
    v: int, n: int, m: int, s: int, L: int, seed: int = 42
) -> dict[int, torch.Tensor]:
    """
    Sample random rules for a random hierarchy model.
    """
    random.seed(seed)
    tuples = list(product(*[range(v) for _ in range(s)]))

    rules: dict[int, torch.Tensor] = {}
    rules[0] = torch.tensor(random.sample(tuples, n * m)).reshape(n, m, -1)
    for i in range(1, L):
        rules[i] = torch.tensor(random.sample(tuples, v * m)).reshape(v, m, -1)
    return rules
