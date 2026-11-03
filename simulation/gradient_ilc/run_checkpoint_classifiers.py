from __future__ import annotations

import argparse
import os
import pickle
import tempfile
import traceback
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

import init
import model
import rhm
def hash_tensor(tensor):
    return hash(tensor.numpy().tobytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MLP classifiers on final checkpoint representations."
    )
    parser.add_argument("--simulation_dir", required=True, type=Path)
    parser.add_argument("--output", default="classifier_results.pkl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--classifier_epochs",
        type=int,
        default=None,
        help="Classifier epochs per checkpoint. Defaults to checkpoint args.Nepoch.",
    )
    parser.add_argument("--classifier_batch_size", type=int, default=128)
    parser.add_argument("--representation_batch_size", type=int, default=512)
    parser.add_argument("--validation_size", type=int, default=3000)
    parser.add_argument("--validation_seed", type=int, default=519823)
    parser.add_argument("--classifier_hidden_dim", type=int, default=128)
    parser.add_argument("--classifier_dropout", type=float, default=0.4)
    parser.add_argument("--classifier_lr", type=float, default=1e-3)
    parser.add_argument("--classifier_weight_decay", type=float, default=1e-2)
    parser.add_argument("--classifier_seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        action="append",
        default=None,
        help="Only process the given checkpoint step. Can be passed multiple times.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Record checkpoint errors and continue instead of failing the process.",
    )
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def find_checkpoints(simulation_dir: Path, steps: set[int] | None) -> list[Path]:
    checkpoint_dir = simulation_dir / "checkpoints"
    checkpoint_paths = sorted(
        checkpoint_dir.glob("checkpoint_step_*.pt"), key=checkpoint_step
    )
    if steps is not None:
        checkpoint_paths = [p for p in checkpoint_paths if checkpoint_step(p) in steps]
    return checkpoint_paths


def output_path_for(simulation_dir: Path, output: str) -> Path:
    path = Path(output)
    if path.is_absolute():
        return path
    return simulation_dir / path


def atomic_pickle_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        tmp_path = Path(handle.name)
        pickle.dump(obj, handle)
    os.replace(tmp_path, path)


def load_student(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    checkpoint_args = argparse.Namespace(**checkpoint["args"])
    parser = init.init_parser()
    init.ensure_args_defaults(checkpoint_args, parser=parser)
    checkpoint_args.device = device
    student = model.init_model(checkpoint_args, checkpoint)
    student.eval()
    return student, checkpoint_args, checkpoint


def obtain_reps(
    student: torch.nn.Module,
    samples: torch.Tensor,
    *,
    args: argparse.Namespace,
    batch_size: int,
) -> torch.Tensor:
    reps: list[torch.Tensor] = []
    loader = DataLoader(samples, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        was_training = student.training
        student.eval()
        for batch in loader:
            batch = batch.to(args.device)
            if hasattr(student, "prepare_batch"):
                batch = student.prepare_batch(batch)
            _, batch_reps = student.compute_losses(batch, return_reps=True)
            reps.append(batch_reps[-1].detach().flatten(start_dim=1).cpu())
        if was_training:
            student.train()
    return torch.cat(reps, dim=0)


def build_validation_set(
    rhm_dataset: dict[str, Any],
    *,
    validation_size: int,
    validation_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    used_hashes = {hash_tensor(string.cpu()) for string in rhm_dataset["features"]}
    validation_strings, validation_labels = rhm.sample_with_replacement(
        2 * validation_size, 0, validation_seed, rhm_dataset["rules"]
    )

    good_strings: list[torch.Tensor] = []
    good_labels: list[torch.Tensor] = []
    for string, label in zip(validation_strings, validation_labels):
        string_hash = hash_tensor(string.cpu())
        if string_hash in used_hashes:
            continue
        used_hashes.add(string_hash)
        good_strings.append(string)
        good_labels.append(label)
        if len(good_strings) >= validation_size:
            break

    if len(good_strings) < validation_size:
        raise RuntimeError(
            f"Only found {len(good_strings)} duplicate-free validation samples; "
            f"requested {validation_size}."
        )

    return torch.stack(good_strings), torch.stack(good_labels).long()


def make_classifier(input_dim: int, args: argparse.Namespace, device: str) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, args.classifier_hidden_dim),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(args.classifier_hidden_dim),
        torch.nn.Dropout(args.classifier_dropout),
        torch.nn.Linear(args.classifier_hidden_dim, args.num_classes),
    ).to(device)


def accuracy(classifier: torch.nn.Module, reps: torch.Tensor, labels: torch.Tensor) -> float:
    classifier.eval()
    with torch.no_grad():
        preds = classifier(reps).argmax(dim=1)
        return (preds == labels).float().mean().item()


def train_classifier(
    train_reps: torch.Tensor,
    train_labels: torch.Tensor,
    validation_reps: torch.Tensor,
    validation_labels: torch.Tensor,
    test_reps: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    args: argparse.Namespace,
    checkpoint_seed_offset: int,
) -> dict[str, Any]:
    torch.manual_seed(args.classifier_seed + checkpoint_seed_offset)

    classifier_args = argparse.Namespace(
        classifier_hidden_dim=args.classifier_hidden_dim,
        classifier_dropout=args.classifier_dropout,
        num_classes=args.num_classes,
    )
    classifier = make_classifier(train_reps.shape[1], classifier_args, args.device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.classifier_lr,
        weight_decay=args.classifier_weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(
        args.classifier_seed + checkpoint_seed_offset
    )
    train_loader = DataLoader(
        TensorDataset(train_reps, train_labels),
        batch_size=args.classifier_batch_size,
        shuffle=True,
        generator=generator,
        drop_last=True,
    )

    val_accs: list[float] = []
    test_accs: list[float] = []
    for _ in range(args.resolved_classifier_epochs):
        classifier.train()
        for batch_reps, batch_labels in train_loader:
            batch_reps = batch_reps.to(args.device)
            batch_labels = batch_labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(batch_reps)
            loss = torch.nn.functional.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()

        validation_acc = accuracy(classifier, validation_reps, validation_labels)
        test_acc = accuracy(classifier, test_reps, test_labels)
        val_accs.append(validation_acc)
        test_accs.append(test_acc)

    best_val_acc = max(val_accs)
    best_val_epoch = max(i for i, acc in enumerate(val_accs) if acc == best_val_acc)
    return {
        "val_accs": val_accs,
        "test_accs": test_accs,
        "best_val_epoch": best_val_epoch,
        "best_val_acc": best_val_acc,
        "test_acc_at_best_val": test_accs[best_val_epoch],
        "final_val_acc": val_accs[-1],
        "final_test_acc": test_accs[-1],
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    cli_args: argparse.Namespace,
    checkpoint_index: int,
) -> dict[str, Any]:
    step = checkpoint_step(checkpoint_path)
    student, checkpoint_args, _ = load_student(checkpoint_path, cli_args.device)
    rhm_dataset = init.init_dataset(checkpoint_args)

    validation_strings, validation_labels = build_validation_set(
        rhm_dataset,
        validation_size=cli_args.validation_size,
        validation_seed=cli_args.validation_seed,
    )

    train_reps = obtain_reps(
        student,
        rhm_dataset["train"],
        args=checkpoint_args,
        batch_size=cli_args.representation_batch_size,
    ).to(cli_args.device)
    test_reps = obtain_reps(
        student,
        rhm_dataset["test"],
        args=checkpoint_args,
        batch_size=cli_args.representation_batch_size,
    ).to(cli_args.device)
    validation_reps = obtain_reps(
        student,
        validation_strings,
        args=checkpoint_args,
        batch_size=cli_args.representation_batch_size,
    ).to(cli_args.device)

    train_labels = rhm_dataset["latents_all_train"][0][:, 0].long().to(cli_args.device)
    test_labels = rhm_dataset["latents_all_test"][0][:, 0].long().to(cli_args.device)
    validation_labels = validation_labels.to(cli_args.device)

    cli_args.num_classes = int(checkpoint_args.rhm_v)
    cli_args.resolved_classifier_epochs = (
        int(cli_args.classifier_epochs)
        if cli_args.classifier_epochs is not None
        else int(checkpoint_args.Nepoch)
    )
    classifier_result = train_classifier(
        train_reps,
        train_labels,
        validation_reps,
        validation_labels,
        test_reps,
        test_labels,
        args=cli_args,
        checkpoint_seed_offset=checkpoint_index,
    )

    return {
        "status": "ok",
        "step": step,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_args": {
            "rhm_L": checkpoint_args.rhm_L,
            "rhm_m": checkpoint_args.rhm_m,
            "rhm_v": checkpoint_args.rhm_v,
            "rhm_s": checkpoint_args.rhm_s,
            "rhm_Ntrain": checkpoint_args.rhm_Ntrain,
            "rhm_Ntest": checkpoint_args.rhm_Ntest,
            "rhm_seed_rules": checkpoint_args.rhm_seed_rules,
            "rhm_seed_sample": checkpoint_args.rhm_seed_sample,
            "Nepoch": checkpoint_args.Nepoch,
            "token_dimension": checkpoint_args.token_dimension,
        },
        "classifier_epochs": cli_args.resolved_classifier_epochs,
        **classifier_result,
    }


def main() -> int:
    args = parse_args()
    simulation_dir = args.simulation_dir
    output_path = output_path_for(simulation_dir, args.output)

    if output_path.exists() and not args.overwrite:
        print(f"Output already exists, skipping: {output_path}", flush=True)
        return 0

    if not simulation_dir.exists():
        raise FileNotFoundError(f"Simulation directory not found: {simulation_dir}")

    steps = set(args.checkpoint_step) if args.checkpoint_step is not None else None
    checkpoints = find_checkpoints(simulation_dir, steps)
    if not checkpoints:
        raise FileNotFoundError(
            f"No matching checkpoint_step_*.pt files found under {simulation_dir / 'checkpoints'}"
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "simulation_dir": str(simulation_dir),
        "classifier_config": {
            "classifier_epochs": args.classifier_epochs,
            "classifier_epochs_default": "checkpoint_args.Nepoch",
            "classifier_batch_size": args.classifier_batch_size,
            "representation_batch_size": args.representation_batch_size,
            "validation_size": args.validation_size,
            "validation_seed": args.validation_seed,
            "classifier_hidden_dim": args.classifier_hidden_dim,
            "classifier_dropout": args.classifier_dropout,
            "classifier_lr": args.classifier_lr,
            "classifier_weight_decay": args.classifier_weight_decay,
            "classifier_seed": args.classifier_seed,
        },
        "checkpoint_results": [],
    }

    for idx, checkpoint_path in enumerate(checkpoints):
        print(f"Processing {checkpoint_path}", flush=True)
        try:
            checkpoint_result = evaluate_checkpoint(
                checkpoint_path, cli_args=args, checkpoint_index=idx
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            checkpoint_result = {
                "status": "error",
                "step": checkpoint_step(checkpoint_path),
                "checkpoint_path": str(checkpoint_path),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        result["checkpoint_results"].append(checkpoint_result)
        atomic_pickle_dump(result, output_path)

    print(f"Wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
