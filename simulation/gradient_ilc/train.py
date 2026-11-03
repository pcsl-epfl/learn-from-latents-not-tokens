from __future__ import annotations

import torch
import torch.optim


def _coerce_losses_to_list(losses):
    if torch.is_tensor(losses):
        return [losses]
    if isinstance(losses, dict):
        return list(losses.values())
    return list(losses)


def add_training_args(parent_parser):
    group = parent_parser.add_argument_group("Training Hyperparameters")
    group.add_argument("--max_training_seconds", type=float, default=0, help="Graceful training cutoff; zero disables")
    group.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    group.add_argument("--batch_size", type=int, default=32, help="Batch size")
    group.add_argument("--Nepoch", type=int, default=100, help="Number of epochs")
    group.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["Adam", "AdamW", "sgd"],
        help="Optimizer",
    )
    group.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    group.add_argument(
        "--ema_alpha", type=float, default=0.0, help="EMA alpha for teacher updates"
    )
    group.add_argument(
        "--use_autojac",
        action="store_true",
        help="Use TorchJD autojac for multi-loss aggregation",
    )
    return parent_parser


def init_optimizer(args, student, resumption_state=None):
    if resumption_state is not None:
        opt = init_optimizer(args, student, resumption_state=None)
        opt.load_state_dict(resumption_state["optimizer"])
        return opt
    if args.optimizer == "Adam":
        return torch.optim.Adam(student.parameters(), lr=args.lr)
    if args.optimizer == "AdamW":
        return torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.optimizer == "sgd":
        return torch.optim.SGD(student.parameters(), lr=args.lr, weight_decay=args.wd)
    raise ValueError(f"Unsupported optimizer type: {args.optimizer}")


def train_model(
    student,
    optimizer,
    train_dataloader,
    Nepoch,
    teacher=None,
    starting_step=0,
    ema_alpha=0.0,
    use_autojac=False,
    rhm_data=None,
    enable_probes=False,
):
    autojac_aggregator = None
    if use_autojac:
        import torchjd

        autojac_aggregator = torchjd.aggregation.UPGrad()

    step = 0
    for n in range(Nepoch):
        for batch in train_dataloader:
            step += 1
            if starting_step >= step:
                continue
            optimizer.zero_grad()
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch_tokens = batch
            device = next(student.parameters()).device
            batch_tokens = batch_tokens.to(device)
            if hasattr(student, "prepare_batch"):
                batch_prepared = student.prepare_batch(batch_tokens)
            else:
                batch_prepared = batch_tokens

            probe_targets = None
            if enable_probes:
                if rhm_data is None or "inverse_rules" not in rhm_data:
                    raise ValueError(
                        "rhm_data with inverse_rules is required for probe training"
                    )
                inv_rules = rhm_data["inverse_rules"]
                rhm_params = rhm_data.get("rhm_params", {})
                rhm_s = int(rhm_params.get("s"))
                rhm_L = int(rhm_params.get("L"))
                # Use CPU for inverse-rule lookups.
                batch_cpu = batch_tokens.detach().cpu()
                if batch_cpu.dim() == 3:
                    batch_cpu = batch_cpu.argmax(dim=1)
                import rhm

                latents = rhm.build_latents_from_inv_rules(
                    batch_cpu, inv_rules, rhm_s, rhm_L
                )
                model_depth = getattr(student, "model_depth", rhm_L - 1)
                probe_targets = [
                    latents[rhm_L - 1 - idx].to(device) for idx in range(model_depth)
                ]
                # print("probe target shapes: ", [pt.shape for pt in probe_targets])

            losses = student.compute_losses(
                batch_prepared, teacher=teacher, probe_targets=probe_targets
            )
            loss_list = _coerce_losses_to_list(losses)
            if use_autojac:
                torchjd.backward(loss_list, autojac_aggregator)
            else:
                total_loss = sum(loss_list)
                total_loss.backward()
            optimizer.step()

            if teacher is not None:
                with torch.no_grad():
                    for param_t, param_m in zip(
                        teacher.parameters(), student.parameters()
                    ):
                        param_t.data.mul_(ema_alpha).add_(
                            (1 - ema_alpha) * param_m.data
                        )

            train_model_state = {"step": step, "epoch": n}
            _ = yield train_model_state
