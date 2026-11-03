from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

import rhm


class ContextPredictor(nn.Module):
    def __init__(
        self,
        s: int,
        v: int,
        h: int,
        h2: int = 0,
        normalization: str = "batchnorm",
        loss: str = "cel",
        device: str = "cpu",
    ):
        super().__init__()
        h2 = h2 if h2 > 0 else h
        self.s = s
        self.v = v

        self.linear1 = nn.Conv1d(v, h, kernel_size=s, stride=s)
        self.linear2 = nn.Conv1d(h, h2, kernel_size=1)
        self.linear3 = nn.Conv1d(h2, s * (s - 1) * s * v, kernel_size=1)

        if normalization == "batchnorm":
            self.norm1 = nn.BatchNorm1d(h)
            self.norm2 = nn.BatchNorm1d(h2)
        elif normalization == "layernorm":
            self.norm1 = nn.LayerNorm(h)
            self.norm2 = nn.LayerNorm(h2)
        else:
            raise ValueError('normalization must be one of ["batchnorm", "layernorm"]')

        if loss not in ["cel", "mse"]:
            raise ValueError('loss must be one of ["cel", "mse"]')
        self.loss = loss
        self.normalization = normalization
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self._apply_norm(x, self.norm1)
        x = F.relu(x)
        x = self.linear2(x)
        x = self._apply_norm(x, self.norm2)
        x = F.relu(x)
        x = self.linear3(x)
        return x

    def _apply_norm(self, x: torch.Tensor, norm_layer: nn.Module) -> torch.Tensor:
        if self.normalization == "layernorm":
            x = x.permute(0, 2, 1)
            x = norm_layer(x)
            x = x.permute(0, 2, 1)
        else:
            x = norm_layer(x)
        return x

    def prediction_pdf(self, prediction: torch.Tensor) -> torch.Tensor:
        prediction = einops.rearrange(
            prediction,
            "batch (s1 s_target_latent s_target_token vocab) i -> "
            "batch vocab s1 s_target_latent s_target_token i",
            s1=self.s,
            s_target_latent=self.s - 1,
            s_target_token=self.s,
        )
        prediction = F.softmax(prediction, dim=1)
        prediction = einops.rearrange(
            prediction,
            "batch vocab s1 s_target_latent s_target_token i -> "
            "batch (s1 s_target_latent s_target_token vocab) i",
            s1=self.s,
            s_target_latent=self.s - 1,
            s_target_token=self.s,
        )
        return prediction

    def layer_loss(
        self,
        x: torch.Tensor,
        prediction: torch.Tensor | None = None,
        xtarget: torch.Tensor | None = None,
    ) -> torch.Tensor:
        xtarget = x if xtarget is None else xtarget
        xtarget = einops.repeat(
            xtarget,
            "batch vocab (s_ell2 s_target_latent s_target_token) -> "
            "batch vocab s_ell2 s1 s_target_latent s_target_token",
            s_target_latent=self.s,
            s_target_token=self.s,
            s1=self.s,
        )
        prediction = prediction if prediction is not None else self.forward(x)
        prediction = einops.rearrange(
            prediction,
            "batch (s1_1 s_target_latent s_target_token vocab) (s_ell2 s1_2) -> "
            "batch vocab s_ell2 s1_1 s1_2 s_target_latent s_target_token",
            s1_1=self.s,
            s1_2=self.s,
            s_target_latent=self.s - 1,
            s_target_token=self.s,
        )
        prediction = prediction[:, :, :, torch.arange(self.s), torch.arange(self.s), :]
        prediction = einops.rearrange(
            prediction,
            "batch vocab s_ell2 s1 s_target_latent s_target_token -> "
            "batch vocab s_ell2 (s1 s_target_latent) s_target_token",
        )
        mask = ~torch.eye(self.s, dtype=torch.bool, device=self.device)
        xtargets_masked = xtarget[:, :, :, mask]

        if self.loss == "mse":
            return F.mse_loss(prediction, xtargets_masked, reduction="mean")
        return F.cross_entropy(prediction, xtargets_masked, reduction="mean")


class ContextClusterer(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        max_classes: int,
        h: int,
        similarity_mode: str = "cosine",
        input_norm: str = "none",
        similarity_margin: float = 0.0,
        similarity_penalty: float = 0.0,
        maximum_comparisons: int = -1,
        subtract_mean: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.linear1 = nn.Conv1d(input_dimension, h, kernel_size=1, stride=1)
        self.norm1 = nn.BatchNorm1d(h)
        self.nonlinearity1 = nn.ReLU()
        self.linear2 = nn.Conv1d(h, max_classes, kernel_size=1, stride=1)
        self.norm2 = nn.BatchNorm1d(max_classes)
        self.nonlinearity2 = nn.Softmax(dim=1)
        self.similarity_mode = similarity_mode
        self.similarity_margin = similarity_margin
        self.similarity_penalty = similarity_penalty
        self.input_norm = input_norm
        self.subtract_mean = subtract_mean
        self.maximum_comparisons = maximum_comparisons
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.nonlinearity1(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = self.nonlinearity2(x)
        return x

    def clustering_loss(
        self, x: torch.Tensor, clusters: torch.Tensor | None = None
    ) -> torch.Tensor:
        clusters = clusters if clusters is not None else self.forward(x)
        clusters = einops.rearrange(clusters, "b d i -> (b i) d")
        x = einops.rearrange(x, "b d i -> (b i) d")

        if self.maximum_comparisons > 0 and x.shape[0] > self.maximum_comparisons:
            indices = torch.randperm(x.shape[0], device=x.device)[
                : self.maximum_comparisons
            ]
            x = x[indices]
            clusters = clusters[indices]

        if self.subtract_mean:
            x = x - x.mean(dim=0, keepdim=True)

        if self.input_norm == "L2":
            x = F.normalize(x, dim=1, p=2)
        elif self.input_norm == "L1":
            x = F.normalize(x, dim=1, p=1)
        elif self.input_norm != "none":
            raise ValueError("Invalid input normalization mode.")

        if self.similarity_mode == "cosine":
            similarity = 0.5 * (
                1 + F.cosine_similarity(x[:, None, :], x[None, :, :], dim=-1)
            )
        else:
            raise ValueError("Invalid similarity mode.")

        D_ij = 1 - similarity
        C_ij = (clusters[:, None, :] - clusters[None, :, :]).pow(2).sum(dim=-1)

        loss_sim = C_ij * F.relu(similarity - self.similarity_margin)
        loss_sep = (clusters[:, None, :] * clusters[None, :, :]).sum(dim=-1) * D_ij
        loss = loss_sim + self.similarity_penalty * loss_sep
        loss = loss[~torch.eye(loss.shape[0], dtype=torch.bool, device=loss.device)]
        return loss.mean()


def cluster_sparsity(cluster_assignments: torch.Tensor) -> torch.Tensor:
    squared = cluster_assignments**2
    sum_squared = squared.sum(dim=1)
    return sum_squared.mean()


@dataclass
class PtCModule:
    predictor: ContextPredictor
    clusterer: ContextClusterer

    def to_module(self) -> nn.Module:
        return nn.ModuleList([self.predictor, self.clusterer])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction = self.predictor(x)
        clusters = self.clusterer(prediction)
        return clusters


class PtC(nn.Module):
    def __init__(
        self,
        *,
        s: int,
        token_dimension: int,
        model_depth: int,
        hidden_dim: int,
        stop_grad_between_modules: bool = False,
        stop_grad_between_clusterer_and_predictor: bool = False,
        cluster_pdfs: bool = False,
        cluster_similarity_mode: str = "cosine",
        cluster_input_norm: str = "none",
        cluster_similarity_margin: float = 0.0,
        cluster_similarity_penalty: float = 0.0,
        cluster_subtract_mean: bool = True,
        cluster_maximum_comparisons: int = -1,
        cluster_sparsity_reward: float = 0.0,
        predictor_norm: str = "batchnorm",
        predictor_loss: str = "cel",
        enable_probes: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.s = s
        self.token_dimension = token_dimension
        self.model_depth = model_depth
        self.stop_grad_between_modules = stop_grad_between_modules
        self.stop_grad_between_clusterer_and_predictor = (
            stop_grad_between_clusterer_and_predictor
        )
        self.cluster_pdfs = cluster_pdfs
        self.cluster_similarity_mode = cluster_similarity_mode
        self.cluster_input_norm = cluster_input_norm
        self.cluster_similarity_margin = cluster_similarity_margin
        self.cluster_similarity_penalty = cluster_similarity_penalty
        self.cluster_subtract_mean = cluster_subtract_mean
        self.cluster_maximum_comparisons = cluster_maximum_comparisons
        self.cluster_sparsity_reward = cluster_sparsity_reward
        self.enable_probes = bool(enable_probes)

        modules = []
        for _ in range(model_depth):
            predictor = ContextPredictor(
                s=s,
                v=token_dimension,
                h=hidden_dim,
                normalization=predictor_norm,
                loss=predictor_loss,
                device=device,
            )
            clusterer = ContextClusterer(
                input_dimension=s**2 * (s - 1) * token_dimension,
                max_classes=token_dimension,
                h=hidden_dim,
                similarity_mode=cluster_similarity_mode,
                input_norm=cluster_input_norm,
                similarity_margin=cluster_similarity_margin,
                similarity_penalty=cluster_similarity_penalty,
                maximum_comparisons=cluster_maximum_comparisons,
                subtract_mean=cluster_subtract_mean,
                device=device,
            )
            modules.append(PtCModule(predictor, clusterer).to_module())
        self.modules_list = nn.ModuleList(modules)
        self.probes = None
        if self.enable_probes:
            self.probes = nn.ModuleList(
                [
                    nn.Conv1d(token_dimension, token_dimension, kernel_size=1)
                    for _ in range(model_depth)
                ]
            )
            self.probe_loss_fn = nn.CrossEntropyLoss()
        self.to(device)

    def prepare_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.dim() == 3:
            return batch
        if batch.max().item() >= self.token_dimension:
            raise ValueError("Input token exceeds token_dimension")
        return rhm.convert_features_to_one_hot(batch, self.token_dimension)

    def _module_forward(
        self,
        module: nn.ModuleList,
        x: torch.Tensor,
        x_t: torch.Tensor | None,
        teacher_module: nn.ModuleList | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        teacher_clusters = None
        teacher_prediction = None
        if teacher_module is not None and x_t is not None:
            with torch.no_grad():
                teacher_pred = teacher_module[0]
                teacher_prediction = teacher_pred(x_t)
                if self.cluster_pdfs:
                    teacher_prediction = teacher_pred.prediction_pdf(teacher_prediction)
                teacher_clusters = teacher_module[1].forward(teacher_prediction)

        pred = module[0]
        clusterer = module[1]
        prediction = pred(x)
        pred_loss = pred.layer_loss(x, prediction, xtarget=x_t)

        if self.stop_grad_between_clusterer_and_predictor:
            prediction = prediction.detach()
        if self.cluster_pdfs:
            prediction = pred.prediction_pdf(prediction)

        prediction_target = (
            teacher_prediction if teacher_prediction is not None else prediction
        )

        clusters_full = clusterer.forward(prediction)
        clusters_loss = clusters_full

        # simple memory cap to reduce pairwise comparisons
        if prediction_target is not None:
            memory_limit = 1e6
            batch_limit = max(
                1, int(memory_limit**0.5 / max(1, prediction_target.shape[2]))
            )
            prediction_target = prediction_target[:batch_limit]
            clusters_loss = clusters_full[:batch_limit]

        clust_loss = clusterer.clustering_loss(prediction_target, clusters_loss)
        if self.cluster_sparsity_reward > 0:
            clust_loss = clust_loss - self.cluster_sparsity_reward * cluster_sparsity(
                clusters_loss
            )
        return pred_loss, clust_loss, clusters_full, teacher_clusters

    def compute_losses(
        self,
        batch: torch.Tensor,
        teacher: "PtC | None" = None,
        return_reps: bool = False,
        probe_targets: list[torch.Tensor] | None = None,
        **_: Any,
    ) -> Any:
        losses: dict[str, torch.Tensor] = {}
        reps: list[torch.Tensor] = []

        x = batch
        x_t = batch if teacher is not None else None

        for idx, module in enumerate(self.modules_list):
            teacher_module = teacher.modules_list[idx] if teacher is not None else None
            pred_loss, clust_loss, clusters, teacher_clusters = self._module_forward(
                module, x, x_t, teacher_module
            )
            losses[f"ptc_layer{idx}_pred_loss"] = pred_loss
            losses[f"ptc_layer{idx}_clust_loss"] = clust_loss
            reps.append(clusters.detach())

            if (
                self.enable_probes
                and probe_targets is not None
                and self.probes is not None
            ):
                targets = probe_targets[idx].to(clusters.device).long()
                probe_input = clusters.detach()
                probe_logits = self.probes[idx](probe_input)
                # print(
                #     f"Probe training index: {idx}",
                #     f"Probe input shape: {probe_input.shape}",
                #     f"probe_logits shape: {probe_logits.shape}",
                #     f"targets shape: {targets.shape}",
                # )
                probe_logits_flat = probe_logits.permute(0, 2, 1).reshape(
                    -1, self.token_dimension
                )
                target_flat = targets.reshape(-1)
                # print(
                #     f"Probe training index: {idx}",
                #     f"probe_logits_flat shape: {probe_logits_flat.shape}",
                #     f"target_flat shape: {target_flat.shape}",
                # )
                losses[f"ptc_layer{idx}_probe_loss"] = self.probe_loss_fn(
                    probe_logits_flat, target_flat
                )

            x = clusters.detach() if self.stop_grad_between_modules else clusters
            x_t = teacher_clusters if teacher is not None else None

        if return_reps:
            return losses, reps
        return losses


def add_ptc_args(parent_parser):
    group = parent_parser.add_argument_group("PtC Parameters")
    group.add_argument(
        "--ptc", action="store_true", help="Use predict-then-cluster (PtC) model"
    )
    group.add_argument(
        "--ptc_hidden_dim",
        type=int,
        default=50,
        help="Hidden dimension for PtC modules",
    )
    group.add_argument(
        "--ptc_normalization",
        type=str,
        default="batchnorm",
        choices=["batchnorm", "layernorm"],
        help="Normalization for PtC predictor",
    )
    group.add_argument(
        "--ptc_loss",
        type=str,
        default="cel",
        choices=["cel", "mse"],
        help="Loss type for PtC predictor",
    )
    group.add_argument(
        "--ptc_cluster_pdfs", action="store_true", help="Cluster prediction PDFs"
    )
    group.add_argument(
        "--ptc_cluster_similarity_mode",
        type=str,
        default="cosine",
        help='Similarity mode for clustering (only "cosine" supported)',
    )
    group.add_argument(
        "--ptc_cluster_input_norm",
        type=str,
        default="none",
        choices=["L2", "L1", "none"],
        help="Input normalization for clustering loss",
    )
    group.add_argument(
        "--ptc_cluster_similarity_margin",
        type=float,
        default=0.0,
        help="Margin applied to similarity scores",
    )
    group.add_argument(
        "--ptc_cluster_similarity_penalty",
        type=float,
        default=0.0,
        help="Penalty to apply to separated assignments",
    )
    group.add_argument(
        "--ptc_cluster_subtract_mean",
        action="store_true",
        help="Subtract mean from inputs before clustering loss",
    )
    group.add_argument(
        "--ptc_cluster_maximum_comparisons",
        type=int,
        default=-1,
        help="Maximum comparisons in clustering loss (-1 for no limit)",
    )
    group.add_argument(
        "--ptc_cluster_sparsity_reward",
        type=float,
        default=0.0,
        help="Reward coefficient for cluster sparsity",
    )
    group.add_argument(
        "--ptc_stop_grad_between_clusterer_and_predictor",
        action="store_true",
        help="Stop gradients between predictor and clusterer",
    )
    return parent_parser


def init_ptc_from_args(args) -> PtC:
    return PtC(
        s=args.rhm_s,
        token_dimension=args.token_dimension,
        model_depth=args.model_depth,
        hidden_dim=args.ptc_hidden_dim,
        stop_grad_between_modules=args.stop_grad_between_modules,
        stop_grad_between_clusterer_and_predictor=args.ptc_stop_grad_between_clusterer_and_predictor,
        cluster_pdfs=args.ptc_cluster_pdfs,
        cluster_similarity_mode=args.ptc_cluster_similarity_mode,
        cluster_input_norm=args.ptc_cluster_input_norm,
        cluster_similarity_margin=args.ptc_cluster_similarity_margin,
        cluster_similarity_penalty=args.ptc_cluster_similarity_penalty,
        cluster_subtract_mean=args.ptc_cluster_subtract_mean,
        cluster_maximum_comparisons=args.ptc_cluster_maximum_comparisons,
        cluster_sparsity_reward=args.ptc_cluster_sparsity_reward,
        predictor_norm=args.ptc_normalization,
        predictor_loss=args.ptc_loss,
        enable_probes=args.enable_probes,
        device=args.device,
    )
