"""Historical RHM sampling functions only; no belief-propagation code."""
import torch
import random
from itertools import product

def sample_rules(v, n, m, s, L, seed=42):
    """
    Sample random rules for a random hierarchy model.

    Args:
        v: The number of values each variable can take (vocabulary size, int).
        n: The number of classes (int).
        m: The number of synonymic lower-level representations (multiplicity, int).
        s: The size of lower-level representations (int).
        L: The number of levels in the hierarchy (int).
        seed: Seed for generating the rules.

    Returns:
        A dictionary containing the rules for each level of the hierarchy.
    """
    random.seed(seed)
    tuples = list(product(*[range(v) for _ in range(s)]))

    rules = {}
    rules[0] = torch.tensor(random.sample(tuples, n * m)).reshape(n, m, -1)
    for i in range(1, L):
        rules[i] = torch.tensor(random.sample(tuples, v * m)).reshape(v, m, -1)

    return rules

def torch_datum_allclasses(seed, rules, v, m, ptr=1):

    L = len(rules)

    g = torch.Generator()
    g.manual_seed(seed)

    # Initialize the dictionary to store the values and messages for each variable
    x_st = {}
    x_st[0] = torch.randint(low=0, high=v, size=(ptr,), generator=g)

    # Loop over the levels of the hierarchy
    for i in range(L):
        # Choose a random rule for each variable in the current level
        chosen_rule = torch.randint(low=0, high=m, size=x_st[i].shape, generator=g)

        # Compute the messages for each variable in the current level
        x_st[i + 1] = rules[i][x_st[i], chosen_rule].flatten(start_dim=1)

    return x_st
