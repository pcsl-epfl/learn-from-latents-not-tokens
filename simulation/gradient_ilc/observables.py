from __future__ import annotations

import torch


def add_observation_args(parent_parser):
    group = parent_parser.add_argument_group("Observation & Logging")
    group.add_argument(
        "--output_filename_prefix", type=str, default="", help="Prefix for output files"
    )
    group.add_argument("--observe_epoch_geom_factor", type=float, default=1.5)
    group.add_argument("--observe_epoch_min", type=int, default=1)
    group.add_argument("--observe_epoch_linear_N", type=int, default=100)
    group.add_argument("--observe_step_geom_factor", type=float, default=1.5)
    group.add_argument("--observe_step_min", type=int, default=100)
    group.add_argument("--observe_step_linear_N", type=int, default=100)
    group.add_argument("--observe_output_path", default="observations.pt", type=str)
    group.add_argument(
        "--observe_N_batches",
        type=int,
        default=1,
        help="Number of batches to average over per observation",
    )
    group = parent_parser.add_argument_group("Checkpoint saving")
    group.add_argument("--disable_checkpoint", action="store_true")
    group.add_argument("--observe_checkpoint_step_geom_factor", type=float, default=1.5)
    group.add_argument("--observe_checkpoint_step_min", type=int, default=1000)
    group.add_argument("--observe_checkpoint_step_linear_N", type=int, default=0)
    group.add_argument("--observe_checkpoint_folder", default="checkpoints", type=str)
    # group = parent_parser.add_argument_group("Observation options")
    # We don't need to worry about observing embeddings, since those are trivial for the rhm.
    # group.add_argument("--observe_embeddings", action="store_true")
    return parent_parser


def _sparsity(z_hat: torch.Tensor) -> float:
    if z_hat.numel() == 0:
        return 0.0
    norm1 = torch.linalg.norm(z_hat, ord=1, dim=1)
    norm2 = torch.linalg.norm(z_hat, ord=2, dim=1)
    n = z_hat.shape[1]
    denom = torch.sqrt(torch.tensor(float(n), device=z_hat.device)) - 1
    sparsity = (
        torch.sqrt(torch.tensor(float(n), device=z_hat.device)) - norm1 / norm2
    ) / denom
    return sparsity.mean().item()


def _cluster_count(labels: torch.Tensor) -> int:
    if labels.numel() == 0:
        return 0
    return int(torch.unique(labels).numel())


def _misclustering(w: torch.Tensor, w_true: torch.Tensor) -> int:
    misclusterings = 0
    for i in torch.unique(w):
        w_trues = w_true[w == i]
        n = w_trues.shape[0]
        if n > 1:
            total_pairs = n * (n - 1) // 2
            _, counts = torch.unique(w_trues, return_counts=True)
            same_label_pairs = (counts * (counts - 1) // 2).sum()
            misclusterings += (total_pairs - same_label_pairs).item()
    return int(misclusterings)


def _hungarian_accuracy(z_estimate: torch.Tensor, z_true: torch.Tensor) -> float:
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    z_est_np = z_estimate.flatten().cpu().numpy()
    z_true_np = z_true.flatten().cpu().numpy()
    if z_est_np.size == 0:
        return 0.0
    n = max(z_est_np.max(), z_true_np.max()) + 1
    conf = np.zeros((n, n), dtype=int)
    for t, p in zip(z_true_np, z_est_np):
        conf[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-conf)
    mapping = {p: t for t, p in zip(row_ind, col_ind)}
    z_aligned = np.array([mapping[p] for p in z_est_np])
    return float((z_aligned == z_true_np).mean())


def geom_and_lin(geom_min, geom_factor, max_val, Nlinear):
    import numpy as np

    max_i = (
        int(np.ceil(np.log(max_val / geom_min) / np.log(geom_factor)))
        if max_val > 0
        else 0
    )
    geom = [int(np.floor(geom_min * geom_factor**i)) for i in range(max_i)]
    linear = []
    if Nlinear > 2:
        linear = [
            int(s) for s in np.round(np.linspace(0, max_val, num=Nlinear)).astype(int)
        ]
    return sorted(set(geom + linear))


def init_observer(
    steps_per_epoch: int,
    train_eval_dataloader,
    test_dataloader,
    val_dataloader,
    args,
    observer_state=None,
    rhm_data=None,
):
    obs = observer(
        train_eval_dataloader,
        test_dataloader,
        val_dataloader,
        args,
        steps_per_epoch,
        rhm_data=rhm_data,
    )
    if observer_state is not None:

        def _field(state, name, default):
            if isinstance(state, dict):
                return state.get(name, default)
            return getattr(state, name, default)

        samples = _field(observer_state, "samples", [])
        sampled_epochs = _field(observer_state, "sampled_epochs", [])
        sampled_steps = _field(observer_state, "sampled_steps", [])

        obs.samples = list(samples) if samples is not None else []
        obs.sampled_epochs = list(sampled_epochs) if sampled_epochs is not None else []
        obs.sampled_steps = list(sampled_steps) if sampled_steps is not None else []
    return obs


class observer:
    def __init__(
        self,
        train_eval_dataloader,
        test_dataloader,
        val_dataloader,
        args,
        steps_per_epoch: int,
        rhm_data=None,
    ):
        import os

        self.args = args
        self.observe_output_path = os.path.join(
            args.output_filename_prefix, args.observe_output_path
        )
        self.observe_checkpoint_folder = os.path.join(
            args.output_filename_prefix, args.observe_checkpoint_folder
        )
        self.test_dataloader = test_dataloader
        self.train_dataloader = train_eval_dataloader
        self.val_dataloader = val_dataloader
        self.rhm_data = rhm_data
        self.observe_N_batches = int(getattr(args, "observe_N_batches", 1))

        max_steps = steps_per_epoch * args.Nepoch
        self.sample_epochs = geom_and_lin(
            args.observe_epoch_min,
            args.observe_epoch_geom_factor,
            args.Nepoch,
            args.observe_epoch_linear_N,
        )
        self.sample_steps = geom_and_lin(
            args.observe_step_min,
            args.observe_step_geom_factor,
            max_steps,
            args.observe_step_linear_N,
        )
        self.checkpoint_steps = geom_and_lin(
            args.observe_checkpoint_step_min,
            args.observe_checkpoint_step_geom_factor,
            max_steps,
            args.observe_checkpoint_step_linear_N,
        )

        self.sampled_epochs = []
        self.sampled_steps = []
        self.samples = []

    def clear_samples_after_step(self, starting_step: int):
        if starting_step is None:
            return
        starting_step = int(starting_step)
        kept_epochs = []
        kept_steps = []
        kept_samples = []
        for epoch, step, sample in zip(
            self.sampled_epochs, self.sampled_steps, self.samples
        ):
            sample_step = sample.get("step", step)
            if sample_step <= starting_step:
                kept_epochs.append(epoch)
                kept_steps.append(step)
                kept_samples.append(sample)
        self.sampled_epochs = kept_epochs
        self.sampled_steps = kept_steps
        self.samples = kept_samples

    def save(self):
        import os

        os.makedirs(os.path.dirname(self.observe_output_path) or ".", exist_ok=True)
        torch.save(self, self.observe_output_path)

    @staticmethod
    def checkpoint_name(path: str, step: int):
        import os

        return os.path.join(path, f"checkpoint_step_{step}.pt")

    def checkpoint(self, student, teacher, optimizer, train_model_state):
        if self.args.disable_checkpoint:
            return
        state = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict() if teacher is not None else None,
            "optimizer": optimizer.state_dict(),
            "train_model_state": train_model_state,
            "args": vars(self.args),
        }
        step = train_model_state["step"]
        fname = observer.checkpoint_name(self.observe_checkpoint_folder, step)
        import os

        os.makedirs(self.observe_checkpoint_folder, exist_ok=True)
        torch.save(state, fname)

    def _next_batch(self, dataloader):
        batch = next(iter(dataloader))
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        return batch

    def observe(self, epoch, step, student, teacher):

        sample = {"epoch": epoch, "step": step}
        device = next(student.parameters()).device

        import rhm

        def _batch_iter(dataloader, n_batches: int):
            it = iter(dataloader)
            for _ in range(n_batches):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(dataloader)
                    batch = next(it)
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                yield batch

        def _extract_tokens(batch: torch.Tensor) -> torch.Tensor:
            if batch.dim() == 3:
                return batch.argmax(dim=1)
            return batch

        def _compute_latents(batch_tokens: torch.Tensor):
            if self.rhm_data is None or "inverse_rules" not in self.rhm_data:
                return None
            inv_rules = self.rhm_data["inverse_rules"]
            rhm_params = self.rhm_data.get("rhm_params", {})
            rhm_s = int(rhm_params.get("s"))
            rhm_L = int(rhm_params.get("L"))
            return rhm.build_latents_from_inv_rules(
                batch_tokens, inv_rules, rhm_s, rhm_L
            )

        def _metrics_for_reps(reps, latents, student_model):
            metrics = []
            rhm_params = self.rhm_data.get("rhm_params", {}) if self.rhm_data else {}
            rhm_L = int(rhm_params.get("L", 0))
            for idx, rep in enumerate(reps):
                gt = latents[rhm_L - 1 - idx]
                pred_labels = rep.argmax(dim=1)
                pred_flat = pred_labels.reshape(-1)
                gt_flat = gt.reshape(-1).to(pred_flat.device)
                rep_flat = rep.permute(0, 2, 1).reshape(-1, rep.shape[1])

                layer_metrics = {
                    "sparsity": _sparsity(rep_flat),
                    "cluster_count": _cluster_count(pred_flat),
                    "misclustering": _misclustering(pred_flat, gt_flat),
                    "accuracy": _hungarian_accuracy(pred_flat, gt_flat),
                }
                if getattr(student_model, "enable_probes", False) and getattr(
                    student_model, "probes", None
                ):
                    probe = student_model.probes[idx]
                    with torch.no_grad():
                        probe_logits = probe(rep.detach())
                    probe_pred = probe_logits.argmax(dim=1)
                    probe_pred_flat = probe_pred.reshape(-1)
                    layer_metrics["probe_accuracy"] = (
                        (probe_pred_flat == gt_flat).float().mean().item()
                    )
                metrics.append(layer_metrics)
            return metrics

        def _accumulate_losses(acc, batch_losses):
            for key, val in batch_losses.items():
                acc[key] = acc.get(key, 0.0) + float(val.detach().cpu().item())

        def _average_losses(acc, n_batches):
            return {k: v / n_batches for k, v in acc.items()}

        def _accumulate_metrics(acc, batch_metrics):
            if not acc:
                for layer_metrics in batch_metrics:
                    acc.append({k: 0.0 for k in layer_metrics})
            for layer_idx, layer_metrics in enumerate(batch_metrics):
                for key, val in layer_metrics.items():
                    acc[layer_idx][key] = acc[layer_idx].get(key, 0.0) + float(val)

        def _average_metrics(acc, n_batches):
            for layer_metrics in acc:
                for key in layer_metrics:
                    layer_metrics[key] = layer_metrics[key] / n_batches
            return acc

        def _observe_split(dataloader):
            if len(dataloader) == 0:
                return {}, []
            loss_acc = {}
            metrics_acc = []
            for batch in _batch_iter(dataloader, self.observe_N_batches):
                batch = batch.to(device)
                batch_tokens = _extract_tokens(batch).detach().cpu()
                latents = _compute_latents(batch_tokens)
                if hasattr(student, "prepare_batch"):
                    batch_prepared = student.prepare_batch(batch)
                else:
                    batch_prepared = batch

                losses, reps = student.compute_losses(
                    batch_prepared, teacher=teacher, return_reps=True
                )
                _accumulate_losses(loss_acc, losses)
                if latents is not None:
                    metrics = _metrics_for_reps(reps, latents, student)
                    _accumulate_metrics(metrics_acc, metrics)

            loss_avg = _average_losses(loss_acc, self.observe_N_batches)
            metrics_avg = (
                _average_metrics(metrics_acc, self.observe_N_batches)
                if metrics_acc
                else []
            )
            return loss_avg, metrics_avg

        with torch.no_grad():
            train_losses, train_metrics = _observe_split(self.train_dataloader)
            test_losses, test_metrics = _observe_split(self.test_dataloader)
            val_losses, val_metrics = _observe_split(self.val_dataloader)

        sample["train_losses"] = train_losses
        sample["test_losses"] = test_losses
        sample["val_losses"] = val_losses
        sample["metrics"] = {
            "train": train_metrics,
            "test": test_metrics,
            "val": val_metrics,
        }

        self.sampled_epochs.append(epoch)
        self.sampled_steps.append(step)
        self.samples.append(sample)

    def get_observable(self, observable: str):
        return [sample[observable] for sample in self.samples]

    def __getitem__(self, key):
        return self.get_observable(key)

    def __len__(self):
        return len(self.samples)

    def __contains__(self, key):
        return all(key in sample for sample in self.samples)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("test_dataloader", None)
        state.pop("train_dataloader", None)
        state.pop("val_dataloader", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.test_dataloader = None
        self.train_dataloader = None
        self.val_dataloader = None
