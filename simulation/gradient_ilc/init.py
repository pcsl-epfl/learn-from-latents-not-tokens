from __future__ import annotations

import argparse
from torch.utils.data import DataLoader, TensorDataset

import rhm


def add_data_args(parent_parser):
    group = parent_parser.add_argument_group("RHM Data Parameters")
    group.add_argument(
        "--rhm_L", type=int, default=3, help="Number of hierarchy layers"
    )
    group.add_argument(
        "--rhm_s", type=int, default=3, help="Branching ratio (tuple size)"
    )
    group.add_argument("--rhm_v", type=int, default=5, help="Vocabulary size")
    group.add_argument(
        "--rhm_n",
        type=int,
        default=0,
        help="Number of top-level classes (0 defaults to rhm_v)",
    )
    group.add_argument(
        "--rhm_m", type=int, default=4, help="Number of synonyms per symbol"
    )
    group.add_argument(
        "--rhm_Ntrain", type=int, default=500, help="Number of training samples"
    )
    group.add_argument(
        "--rhm_Ntest", type=int, default=500, help="Number of test samples"
    )
    group.add_argument(
        "--rhm_Nval", type=int, default=0, help="Number of validation samples"
    )
    group.add_argument(
        "--rhm_seed_rules", type=int, default=0, help="Seed for rule sampling"
    )
    group.add_argument(
        "--rhm_seed_sample", type=int, default=10, help="Seed for data sampling"
    )
    group.add_argument(
        "--rhm_sample_without_replacement",
        action="store_true",
        help="Use sampling without replacement",
    )
    group.add_argument(
        "--observe_batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation/observations",
    )
    group.add_argument(
        "--cache_on_device",
        action="store_true",
        help="Cache the dataset on the device ",
    )
    group.add_argument("--seed_torch", type=int, default=42, help="Seed for torch RNG")
    return parent_parser


def init_parser():
    parser = argparse.ArgumentParser()
    parser = add_data_args(parser)
    group = parser.add_argument_group("Resuming training args")
    group.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )
    group.add_argument("--resume_checkpoint_path", type=str, default=None)
    group.add_argument("--resume_observables_path", type=str, default=None)

    import train
    import observables
    import model

    parser = train.add_training_args(parser)
    parser = observables.add_observation_args(parser)
    parser = model.add_model_args(parser)
    return parser


def ensure_args_defaults(args, parser=None):
    if parser is None:
        parser = init_parser()
    default_args = parser.parse_args([])
    for key, value in vars(default_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def init_dataset(args):
    n = args.rhm_n if args.rhm_n > 0 else args.rhm_v
    rules = rhm.sample_rules(
        args.rhm_v, n, args.rhm_m, args.rhm_s, args.rhm_L, seed=args.rhm_seed_rules
    )
    ntrain = int(args.rhm_Ntrain)
    ntest = int(args.rhm_Ntest)
    nval = int(args.rhm_Nval)
    Ntot = ntrain + ntest + nval

    if args.rhm_sample_without_replacement:
        max_data = rhm.compute_pmax(n, args.rhm_m, args.rhm_s, args.rhm_L)
        if Ntot > max_data:
            print(
                "Warning: requested more data than max possible with sampling without replacement. Reducing dataset size to max possible: ",
                max_data,
            )
            ntest = max(0, max_data - ntrain)
            nval = max(0, max_data - ntrain - ntest)
        features, labels = rhm.sample_without_replacement(
            max_data, Ntot, 0, args.rhm_seed_sample, rules
        )
    else:
        features, labels = rhm.sample_with_replacement(
            Ntot, 0, args.rhm_seed_sample, rules
        )

    inv_rules = rhm.build_inverse_rules(rules, s=args.rhm_s, v=args.rhm_v)
    latents_all = rhm.build_latents_from_inv_rules(
        features, inv_rules, args.rhm_s, args.rhm_L
    )

    train_slice = slice(0, ntrain)
    test_slice = slice(ntrain, ntrain + ntest)
    val_slice = slice(ntrain + ntest, ntrain + ntest + nval)

    latents_all_train = {k: v[train_slice] for k, v in latents_all.items()}
    latents_all_test = {k: v[test_slice] for k, v in latents_all.items()}
    latents_all_val = {k: v[val_slice] for k, v in latents_all.items()}

    return {
        "rules": rules,
        "features": features,
        "labels": labels,
        "latents_all": latents_all,
        "inverse_rules": inv_rules,
        "train": features[train_slice],
        "test": features[test_slice],
        "val": features[val_slice],
        "latents_all_train": latents_all_train,
        "latents_all_test": latents_all_test,
        "latents_all_val": latents_all_val,
        "rhm_params": {
            "L": args.rhm_L,
            "s": args.rhm_s,
            "v": args.rhm_v,
            "m": args.rhm_m,
            "n": n,
        },
    }


def init_data_loaders(args, rhm_data):
    train_dataset = TensorDataset(
        rhm_data["train"].to(args.device) if args.cache_on_device else rhm_data["train"]
    )
    test_dataset = TensorDataset(
        rhm_data["test"].to(args.device) if args.cache_on_device else rhm_data["test"]
    )
    val_dataset = TensorDataset(
        rhm_data["val"].to(args.device) if args.cache_on_device else rhm_data["val"]
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.observe_batch_size, shuffle=False
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.observe_batch_size, shuffle=False
    )
    train_eval_dataloader = DataLoader(
        train_dataset, batch_size=args.observe_batch_size, shuffle=False
    )
    return train_dataloader, test_dataloader, val_dataloader, train_eval_dataloader
