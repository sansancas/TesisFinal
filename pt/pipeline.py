#!/usr/bin/env python
"""
Pipeline principal para entrenamiento y evaluación en PyTorch.
Mantiene la estructura del script original basado en TensorFlow pero usando
modelos y utilidades migradas en ``tpt``.
"""
from __future__ import annotations

import csv
import contextlib
import json
import math
import pickle
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import (
    DatasetBundle,
    _ensure_probability_matrix,
    _flatten_labels_to_frames,
    _window_level_labels,
    build_dataloader,
    build_torch_dataset_from_arrays,
    build_windows_dataset,
    collect_records_for_split,
    prepare_model_inputs,
    resolve_cache_target,
    summarize_dataset_bundle,
)
from models.Hybrid import HybridClassifier, HybridOutputs, build_hybrid
from models.TCN import TCNClassifier, TCNOutputs, build_tcn
from utils import (
    DEFAULT_CONFIG_FILENAME,
    PipelineConfig,
    TFRecordExportConfig,
    FoldResult,
    apply_preprocess_config,
    config_to_dict,
    config_summary_df,
    load_config,
    resolve_checkpoint_dir,
    resolve_epoch_time_log_path,
    resolve_preprocess_settings,
    save_cv_outputs,
    validate_config,
)


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return value


# =============================================================================
# Utilidades de entrenamiento
# =============================================================================


class EpochTimeLogger:
    def __init__(self, log_path: Path, context: dict[str, object] | None = None) -> None:
        self.log_path = Path(log_path)
        self.context = dict(context or {})
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.log_path.exists():
            try:
                self.log_path.unlink()
            except OSError:
                pass
        self._fieldnames = ["timestamp", "epoch", "duration_sec", "metrics_json", "context_json"]
        with self.log_path.open("w", encoding="utf-8", newline="") as fp:
            csv.DictWriter(fp, fieldnames=self._fieldnames).writeheader()
        self._start_time: float | None = None

    def on_epoch_begin(self) -> None:
        self._start_time = time.perf_counter()

    def on_epoch_end(
        self,
        epoch: int,
        metrics: dict[str, object] | None = None,
        extra_context: dict[str, object] | None = None,
    ) -> float | None:
        if self._start_time is None:
            return None
        duration = time.perf_counter() - self._start_time
        timestamp = datetime.now(timezone.utc).isoformat()
        metrics_json = json.dumps(metrics or {}, ensure_ascii=False, sort_keys=True, default=_json_default)
        row_context = dict(self.context)
        if extra_context:
            row_context.update(extra_context)
        context_json = json.dumps(row_context, ensure_ascii=False, sort_keys=True, default=_json_default)
        with self.log_path.open("a", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=self._fieldnames)
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "epoch": epoch + 1,
                    "duration_sec": round(duration, 6),
                    "metrics_json": metrics_json,
                    "context_json": context_json,
                }
            )
            fp.flush()
        self._start_time = None
        return duration


class MaxTimeStopping:
    def __init__(self, max_duration_seconds: float, *, verbose: int = 0) -> None:
        if max_duration_seconds <= 0:
            raise ValueError("'max_duration_seconds' debe ser > 0.")
        self.max_duration_seconds = float(max_duration_seconds)
        self.verbose = int(verbose)
        self._start_time: float | None = None
        self._stop_triggered = False

    def on_train_begin(self) -> None:
        self._start_time = time.perf_counter()
        self._stop_triggered = False

    def should_stop(self) -> bool:
        if self._start_time is None or self._stop_triggered:
            return False
        elapsed = time.perf_counter() - self._start_time
        if elapsed >= self.max_duration_seconds:
            self._stop_triggered = True
            if self.verbose:
                minutes = self.max_duration_seconds / 60.0
                print(
                    f"   ↳ MaxTimeStopping detuvo el entrenamiento tras {minutes:.2f} min (límite alcanzado)."
                )
        return self._stop_triggered


class EarlyStoppingState:
    def __init__(self, patience: int, *, mode: str = "max") -> None:
        self.patience = max(1, int(patience))
        self.mode = mode
        self.best: float | None = None
        self.counter = 0

    def update(self, value: float) -> bool:
        improved: bool
        if self.best is None:
            improved = True
        else:
            if self.mode == "max":
                improved = value > self.best + 1e-6
            else:
                improved = value < self.best - 1e-6
        if improved:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        return False

    def should_stop(self) -> bool:
        return self.counter >= self.patience


class MetricCheckpoint:
    def __init__(self, filepath: Path, metric: str, *, mode: str = "max") -> None:
        self.filepath = Path(filepath)
        self.metric = metric
        self.mode = mode
        self.best: float | None = None

    def update(self, model: nn.Module, metrics: dict[str, float]) -> None:
        value = metrics.get(self.metric)
        if value is None:
            return
        improved: bool
        if self.best is None:
            improved = True
        else:
            if self.mode == "max":
                improved = value > self.best + 1e-6
            else:
                improved = value < self.best - 1e-6
        if improved:
            self.best = value
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.filepath)


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float, beta: float, gamma: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-7
        probs = torch.sigmoid(logits)
        targets = targets.float()
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        true_pos = torch.sum(probs * targets, dim=1)
        false_pos = torch.sum(probs * (1.0 - targets), dim=1)
        false_neg = torch.sum((1.0 - probs) * targets, dim=1)
        denominator = true_pos + self.alpha * false_pos + self.beta * false_neg + epsilon
        tversky_index = (true_pos + epsilon) / denominator
        loss = (1.0 - tversky_index) ** self.gamma
        return loss.mean()


def binary_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float,
    alpha: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    epsilon = 1e-7
    probs = torch.sigmoid(logits)
    targets = targets.float()
    pt = torch.where(targets >= 0.5, probs, 1.0 - probs)
    alpha_factor = torch.where(targets >= 0.5, alpha, 1.0 - alpha)
    loss = -(alpha_factor * ((1.0 - pt) ** gamma) * torch.log(pt.clamp(min=epsilon)))
    if weights is not None:
        loss = loss * weights
    return loss.mean()


class LossComputer:
    def __init__(self, config: PipelineConfig) -> None:
        self.loss_type = config.loss_type
        self.alpha = config.focal_alpha
        self.gamma = config.focal_gamma
        self.tversky_alpha = config.tversky_alpha
        self.tversky_beta = config.tversky_beta
        self.tversky_gamma = config.tversky_gamma

    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.loss_type == "binary_crossentropy":
            return F.binary_cross_entropy_with_logits(logits, targets.float(), weight=weights, reduction="mean")
        if self.loss_type == "focal":
            return binary_focal_loss(
                logits,
                targets,
                gamma=self.gamma,
                alpha=self.alpha,
                weights=weights,
            )
        if self.loss_type == "tversky":
            loss = TverskyLoss(self.tversky_alpha, self.tversky_beta, gamma=1.0)
            return loss(logits, targets)
        if self.loss_type == "tversky_focal":
            loss = TverskyLoss(self.tversky_alpha, self.tversky_beta, gamma=self.tversky_gamma)
            return loss(logits, targets)
        raise ValueError(f"Tipo de pérdida no soportado: {self.loss_type}")


def create_loss_factory(config: PipelineConfig) -> Callable[[], LossComputer]:
    return lambda: LossComputer(config)


def describe_loss(config: PipelineConfig) -> str:
    if config.loss_type == "binary_crossentropy":
        return "Binary Crossentropy"
    if config.loss_type == "focal":
        return f"Focal BCE (gamma={config.focal_gamma:g}, alpha={config.focal_alpha:g})"
    if config.loss_type == "tversky":
        return f"Tversky (alpha={config.tversky_alpha:g}, beta={config.tversky_beta:g})"
    if config.loss_type == "tversky_focal":
        return (
            f"Focal Tversky (alpha={config.tversky_alpha:g}, beta={config.tversky_beta:g}, "
            f"gamma={config.tversky_gamma:g})"
        )
    return config.loss_type


def create_optimizer_factory(config: PipelineConfig) -> Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer]:
    optimizer_name = config.optimizer.lower()

    def factory(params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
        if optimizer_name == "adam":
            return torch.optim.Adam(params, lr=config.learning_rate)
        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                params,
                lr=config.learning_rate,
                weight_decay=config.optimizer_weight_decay,
            )
        raise ValueError(f"Optimizador no soportado: {optimizer_name}")

    return factory


def create_optimizer(config: PipelineConfig, model: nn.Module) -> torch.optim.Optimizer:
    factory = create_optimizer_factory(config)
    return factory(model.parameters())


def describe_optimizer(config: PipelineConfig) -> str:
    label = config.optimizer.upper()
    details: list[str] = []
    if config.optimizer == "adamw" and config.optimizer_weight_decay > 0:
        details.append(f"wd={config.optimizer_weight_decay:g}")
    if config.optimizer_use_ema:
        details.append(f"EMA({config.optimizer_ema_momentum:.3f})")
    if details:
        label += f" [{', '.join(details)}]"
    return f"{label} | lr={config.learning_rate:g}"


def create_scheduler_callbacks(
    *,
    has_validation: bool,
    patience: int,
    min_lr: float,
    lr_schedule_type: str,
    learning_rate: float,
    cosine_period: int,
    cosine_min_lr: float | None,
) -> dict[str, object]:
    monitor_metric = "pr_auc" if has_validation else "loss"
    mode = "max" if has_validation else "min"

    lr_schedule_type = lr_schedule_type.lower()
    target_min_lr = cosine_min_lr if cosine_min_lr is not None else min_lr

    def scheduler_factory(optimizer: torch.optim.Optimizer):
        if lr_schedule_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=0.5,
                patience=max(1, patience // 2),
                min_lr=min_lr,
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, cosine_period),
            eta_min=target_min_lr,
        )

    return {
        "monitor": monitor_metric,
        "mode": mode,
        "scheduler_factory": scheduler_factory,
        "patience": patience,
    }


def create_metric_checkpoint_callbacks(
    base_dir: Path,
    *,
    metrics: Sequence[str],
    has_validation: bool,
    prefix: str = "",
) -> list[MetricCheckpoint]:
    base_dir.mkdir(parents=True, exist_ok=True)
    callbacks: list[MetricCheckpoint] = []
    mode = "max" if has_validation else "min"
    for metric in metrics:
        filepath = base_dir / f"{prefix}{metric}_best.pt"
        callbacks.append(MetricCheckpoint(filepath, metric, mode=mode))
    return callbacks


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.any(y_true == 1):
        return float(average_precision_score(y_true, y_score))
    return float("nan")


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size > 1:
        return float(roc_auc_score(y_true, y_score))
    return float("nan")


def make_model(
    *,
    model_type: str,
    input_shape: tuple[int, int],
    feat_dim: int | None,
    num_filters: int,
    kernel_size: int,
    dropout_rate: float,
    rnn_units: int,
    time_step: bool,
) -> nn.Module:
    common_kwargs = dict(
        input_shape=input_shape,
        num_classes=1,
        one_hot=False,
    )
    if model_type == "tcn":
        return build_tcn(
            **common_kwargs,
            num_filters=num_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            time_step_classification=time_step,
            feat_input_dim=feat_dim,
        )
    if model_type == "hybrid":
        return build_hybrid(
            **common_kwargs,
            num_filters=num_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            time_step=time_step,
            rnn_units=rnn_units,
            feat_input_dim=feat_dim,
        )
    raise ValueError(f"Modelo no soportado: {model_type}")


def _prepare_batch(
    batch: tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    inputs, labels = batch

    if isinstance(inputs, list):
        if len(inputs) == 2 and all(isinstance(elem, torch.Tensor) for elem in inputs):
            inputs = (inputs[0], inputs[1])
        elif len(inputs) == 2 and all(isinstance(elem, np.ndarray) for elem in inputs):
            inputs = (inputs[0], inputs[1])

    def _to_tensor(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, np.ndarray):
            return torch.as_tensor(value, device=device)
        if isinstance(value, (list, tuple)):
            if not value:
                return torch.empty(0, device=device)
            if all(isinstance(elem, torch.Tensor) for elem in value):
                return torch.stack([elem.to(device) for elem in value], dim=0)
            if all(isinstance(elem, np.ndarray) for elem in value):
                return torch.as_tensor(np.stack(value, axis=0), device=device)
            return torch.as_tensor(value, device=device)
        return torch.as_tensor(value, device=device)

    if isinstance(inputs, tuple):
        seq, feat = inputs
        seq = _to_tensor(seq)
        feat = _to_tensor(feat)
    else:
        seq = _to_tensor(inputs)
        feat = None

    labels = _to_tensor(labels)
    if labels.dim() == 1:
        labels = labels.unsqueeze(-1)
    return seq, feat, labels


def _compute_class_weights(
    labels: np.ndarray,
    *,
    label_mode: str,
    use_class_weights: bool,
) -> dict[int, float] | None:
    if not use_class_weights:
        return None
    frames = _flatten_labels_to_frames(labels, label_mode=label_mode)
    if frames.size == 0:
        return None
    pos_frac = float(frames.mean())
    if pos_frac <= 0 or pos_frac >= 1:
        print(f"   ↳ Solo una clase en train (pos_frac={pos_frac:.3f}); sin class_weight.")
        return None
    class_weight = {0: 0.5 / (1.0 - pos_frac), 1: 0.5 / pos_frac}
    print(f"   ↳ Balance etiquetas train (frames): {pos_frac:.3f} positivos -> class_weight {class_weight}")
    return class_weight


def _weights_tensor(
    labels: torch.Tensor,
    class_weight: dict[int, float] | None,
) -> torch.Tensor | None:
    if class_weight is None:
        return None
    w0 = class_weight.get(0, 1.0)
    w1 = class_weight.get(1, 1.0)
    weight_tensor = torch.where(labels > 0.5, torch.as_tensor(w1, device=labels.device), torch.as_tensor(w0, device=labels.device))
    return weight_tensor


def _forward_model(
    model: nn.Module,
    inputs: torch.Tensor,
    feat: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    outputs = model(inputs, feat) if isinstance(model, (TCNClassifier, HybridClassifier)) else model(inputs)
    if isinstance(outputs, (TCNOutputs, HybridOutputs)):
        logits = outputs.logits
        aux_losses = outputs.aux_losses
    else:
        logits = outputs
        aux_losses = {}
    return logits, aux_losses


def _accumulate_aux_losses(aux_losses: dict[str, torch.Tensor]) -> torch.Tensor | None:
    if not aux_losses:
        return None
    total = None
    for tensor in aux_losses.values():
        total = tensor if total is None else total + tensor
    return total


def _model_probabilities(logits: torch.Tensor) -> torch.Tensor:
    if logits.size(-1) == 1:
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=-1)


def _collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probs_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            seq, feat, _ = _prepare_batch(batch, device)
            outputs = model(seq, feat) if isinstance(model, (TCNClassifier, HybridClassifier)) else model(seq)
            if isinstance(outputs, (TCNOutputs, HybridOutputs)):
                probs = outputs.probabilities
            else:
                probs = torch.sigmoid(outputs)
            probs_list.append(probs.detach().cpu().numpy())
    return np.concatenate(probs_list, axis=0)


def _compute_evaluation_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    label_mode: str,
) -> dict[str, float]:
    prob_matrix = _ensure_probability_matrix(probs)
    prob_matrix = prob_matrix.reshape(prob_matrix.shape[0], -1)
    frame_labels = _flatten_labels_to_frames(labels, label_mode=label_mode)
    frame_labels = frame_labels.reshape(frame_labels.shape[0], -1)
    if prob_matrix.shape[0] != frame_labels.shape[0]:
        raise RuntimeError("Dimensiones incompatibles entre predicciones y etiquetas.")
    y_flat = frame_labels.ravel().astype(int)
    prob_flat = prob_matrix.ravel()
    pred_flat = (prob_flat >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_flat, pred_flat)),
        "precision": float(precision_score(y_flat, pred_flat, zero_division=0)),
        "recall": float(recall_score(y_flat, pred_flat, zero_division=0)),
        "f1": float(f1_score(y_flat, pred_flat, zero_division=0)),
        "pr_auc": safe_average_precision(y_flat, prob_flat),
        "roc_auc": safe_roc_auc(y_flat, prob_flat),
    }
    return metrics


def _window_predictions(
    probs: np.ndarray,
    labels: np.ndarray,
    patients: np.ndarray,
    records: np.ndarray,
    *,
    label_mode: str,
    fold_idx: int | None = None,
) -> pd.DataFrame:
    prob_matrix = _ensure_probability_matrix(probs)
    window_prob = prob_matrix.mean(axis=1)
    window_true = _window_level_labels(labels, label_mode=label_mode)
    window_pred = (window_prob >= 0.5).astype(int)
    data = {
        "patient": patients,
        "record": records,
        "y_true": window_true.astype(int),
        "y_pred": window_pred,
        "y_prob": window_prob,
    }
    if fold_idx is not None:
        data["fold"] = np.repeat(int(fold_idx), window_true.shape[0])
    return pd.DataFrame(data)


def _train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_computer: LossComputer,
    *,
    device: torch.device,
    class_weight: dict[int, float] | None = None,
    log_interval: int | None = None,
    progress_prefix: str = "",
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    progress_total = None
    if log_interval is not None:
        try:
            progress_total = len(dataloader)
        except TypeError:
            progress_total = None
    progress_printed = False
    for batch_idx, batch in enumerate(dataloader, start=1):
        optimizer.zero_grad(set_to_none=True)
        seq, feat, labels = _prepare_batch(batch, device)
        logits, aux_losses = _forward_model(model, seq, feat)
        logits = logits.view_as(labels)
        weights = _weights_tensor(labels, class_weight)
        primary_loss = loss_computer(logits, labels, weights=weights)
        aux_loss = _accumulate_aux_losses(aux_losses)
        loss = primary_loss + aux_loss if aux_loss is not None else primary_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_batches += 1
        if (
            log_interval is not None
            and progress_total not in (None, 0)
            and (batch_idx % log_interval == 0 or batch_idx == progress_total)
        ):
            progress = batch_idx / progress_total
            print(
                f"{progress_prefix}batches {batch_idx}/{progress_total} ({progress * 100:.0f}%) ",
                end="\r",
                flush=True,
            )
            progress_printed = True
    if progress_printed:
        print()
    return total_loss / max(1, total_batches)


def _evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    label_mode: str,
) -> tuple[float, dict[str, float], np.ndarray]:
    model.eval()
    losses: list[float] = []
    probs_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            seq, feat, labels = _prepare_batch(batch, device)
            logits, aux_losses = _forward_model(model, seq, feat)
            logits = logits.view_as(labels)
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            aux_loss = _accumulate_aux_losses(aux_losses)
            if aux_loss is not None:
                loss = loss + aux_loss
            losses.append(float(loss.detach().cpu()))
            probs = _model_probabilities(logits)
            probs_list.append(probs.detach().cpu().numpy())
    avg_loss = float(np.mean(losses)) if losses else 0.0
    probs = np.concatenate(probs_list, axis=0) if probs_list else np.zeros((0, 1), dtype=np.float32)
    metrics = _compute_evaluation_metrics(probs, dataloader.dataset.labels, label_mode=label_mode)  # type: ignore[arg-type]
    metrics["loss"] = avg_loss
    return avg_loss, metrics, probs


def run_group_cv(
    data: DatasetBundle,
    *,
    model_type: str,
    batch_size: int,
    epochs: int,
    num_filters: int,
    kernel_size: int,
    dropout_rate: float,
    rnn_units: int,
    time_step: bool,
    patience: int,
    min_lr: float,
    folds: int,
    random_seed: int,
    verbose: int = 1,
    optimizer_factory: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer] | None = None,
    optimizer_description: str | None = None,
    loss_factory: Callable[[], LossComputer] | None = None,
    loss_description: str | None = None,
    use_class_weights: bool = True,
    epoch_time_log_path: Path | None = None,
    epoch_time_log_context: dict[str, object] | None = None,
    lr_schedule_type: str = "plateau",
    cosine_annealing_period: int = 10,
    cosine_annealing_min_lr: float | None = None,
    base_learning_rate: float = 1e-3,
    save_metric_checkpoints: bool = True,
    checkpoint_dir: Path | None = None,
    verbose_level: int = 1,
    max_training_minutes: float | None = None,
) -> tuple[list[FoldResult], pd.DataFrame]:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    sequences = data.sequences
    features = data.features
    labels = data.labels.astype(np.float32)
    label_mode = getattr(data, "label_mode", "window")
    feature_names = list(getattr(data, "feature_names", []))
    patients = data.patients
    records = data.records

    if time_step and label_mode != "time_step":
        raise ValueError("El dataset no contiene etiquetas por frame necesarias para 'time_step'.")
    if not time_step and label_mode == "time_step":
        print("⚠️  Advertencia: etiquetas por frame detectadas pero el modelo entrenará a nivel ventana.")

    unique_patients = np.unique(patients).size
    n_splits = max(2, min(folds, unique_patients))
    if n_splits < 2:
        raise RuntimeError("Se requieren al menos 2 pacientes distintos para CV agrupada.")

    input_shape = sequences.shape[1:]
    feat_dim = features.shape[1] if features is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    group_kfold = GroupKFold(n_splits=n_splits)
    labels_for_split = _window_level_labels(labels, label_mode=label_mode)
    fold_results: list[FoldResult] = []
    all_predictions: list[pd.DataFrame] = []

    if optimizer_description:
        print(f"\n⚙️  Optimizador CV: {optimizer_description}")
    if loss_description:
        print(f"   ↳ Pérdida: {loss_description}")

    scheduler_cfg = create_scheduler_callbacks(
        has_validation=True,
        patience=patience,
        min_lr=min_lr,
        lr_schedule_type=lr_schedule_type,
        learning_rate=base_learning_rate,
        cosine_period=cosine_annealing_period,
        cosine_min_lr=cosine_annealing_min_lr,
    )

    epoch_context = dict(epoch_time_log_context or {})
    epoch_context.setdefault("phase", "cv")
    epoch_timer = EpochTimeLogger(epoch_time_log_path, epoch_context) if epoch_time_log_path else None

    for fold_idx, (train_idx, val_idx) in enumerate(
        group_kfold.split(sequences, labels_for_split, groups=patients),
        start=1,
    ):
        print(
            f"\n🔁 Fold {fold_idx}/{n_splits} | pacientes train={np.unique(patients[train_idx]).size}, "
            f"val={np.unique(patients[val_idx]).size}"
        )

        X_train = sequences[train_idx]
        y_train = labels[train_idx]
        X_val = sequences[val_idx]
        y_val = labels[val_idx]

        if features is not None:
            scaler = StandardScaler()
            X_train_feat = scaler.fit_transform(features[train_idx])
            X_val_feat = scaler.transform(features[val_idx])
        else:
            X_train_feat = X_val_feat = None

        model = make_model(
            model_type=model_type,
            input_shape=input_shape,
            feat_dim=feat_dim,
            num_filters=num_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            rnn_units=rnn_units,
            time_step=time_step,
        ).to(device)

        optimizer = (
            optimizer_factory(model.parameters())
            if optimizer_factory is not None
            else torch.optim.Adam(model.parameters(), lr=base_learning_rate)
        )
        scheduler = scheduler_cfg["scheduler_factory"](optimizer)  # type: ignore[index]
        early_stopping = EarlyStoppingState(patience, mode="max")
        loss_computer = loss_factory() if loss_factory is not None else LossComputer(PipelineConfig())

        train_dataset = build_torch_dataset_from_arrays(
            (X_train, X_train_feat) if X_train_feat is not None else X_train,
            y_train,
            label_mode=label_mode,
        )
        val_dataset = build_torch_dataset_from_arrays(
            (X_val, X_val_feat) if X_val_feat is not None else X_val,
            y_val,
            label_mode=label_mode,
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        log_interval = None
        if verbose_level:
            try:
                total_batches = len(train_loader)
            except TypeError:
                total_batches = None
            else:
                if total_batches:
                    log_interval = max(1, total_batches // 5)

        class_weight = _compute_class_weights(y_train, label_mode=label_mode, use_class_weights=use_class_weights)

        stopper = MaxTimeStopping(max_training_minutes * 60.0, verbose=verbose_level) if max_training_minutes else None
        if stopper is not None:
            stopper.on_train_begin()

        best_state = None
        best_metric = -float("inf")

        checkpoint_callbacks: list[MetricCheckpoint] | None = None
        if save_metric_checkpoints and checkpoint_dir is not None:
            fold_dir = checkpoint_dir / f"fold_{fold_idx:02d}"
            checkpoint_callbacks = create_metric_checkpoint_callbacks(
                fold_dir,
                metrics=("pr_auc", "roc_auc", "precision"),
                has_validation=True,
                prefix=f"fold{fold_idx:02d}_",
            )

        for epoch in range(epochs):
            if epoch_timer is not None:
                epoch_timer.on_epoch_begin()

            train_loss = _train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_computer,
                device=device,
                class_weight=class_weight,
                log_interval=log_interval,
                progress_prefix=f"   ↻ Fold {fold_idx:02d} Epoch {epoch + 1:03d}/{epochs}: ",
            )

            train_eval_loss, train_metrics, _ = _evaluate_model(
                model,
                train_eval_loader,
                device=device,
                label_mode=label_mode,
            )
            train_metrics["loss"] = train_eval_loss

            _, val_metrics, _ = _evaluate_model(
                model,
                val_loader,
                device=device,
                label_mode=label_mode,
            )
            monitor_metric = val_metrics.get("pr_auc", 0.0)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitor_metric)
            else:
                scheduler.step()

            epoch_metrics_for_log: dict[str, object] = {"train_loss": train_loss}
            epoch_metrics_for_log.update(
                {
                    ("train_eval_loss" if key == "loss" else f"train_{key}"): value
                    for key, value in train_metrics.items()
                }
            )
            epoch_metrics_for_log.update({f"val_{key}": value for key, value in val_metrics.items()})
            if epoch_timer is not None:
                epoch_timer.on_epoch_end(epoch, metrics=epoch_metrics_for_log, extra_context={"fold": fold_idx})

            if verbose:
                train_summary = (
                    f"train_loss={train_loss:.4f}, "
                    f"train_acc={train_metrics.get('accuracy', float('nan')):.4f}, "
                    f"train_prec={train_metrics.get('precision', float('nan')):.4f}, "
                    f"train_rec={train_metrics.get('recall', float('nan')):.4f}, "
                    f"train_auc={train_metrics.get('roc_auc', float('nan')):.4f}"
                )
                val_summary = (
                    f"val_loss={val_metrics.get('loss', float('nan')):.4f}, "
                    f"val_acc={val_metrics.get('accuracy', float('nan')):.4f}, "
                    f"val_prec={val_metrics.get('precision', float('nan')):.4f}, "
                    f"val_rec={val_metrics.get('recall', float('nan')):.4f}, "
                    f"val_auc={val_metrics.get('roc_auc', float('nan')):.4f}, "
                    f"val_pr_auc={val_metrics.get('pr_auc', float('nan')):.4f}"
                )
                print(
                    f"   Epoch {epoch + 1:03d}/{epochs}: {train_summary}, {val_summary}"
                )

            improved = early_stopping.update(monitor_metric)
            if improved:
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_metric = monitor_metric

            if checkpoint_callbacks is not None:
                for cb in checkpoint_callbacks:
                    cb.update(model, val_metrics)

            if early_stopping.should_stop():
                print(f"   ↳ Early stopping activado (best_pr_auc={best_metric:.4f}).")
                break
            if stopper is not None and stopper.should_stop():
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        val_loader_final = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        probs = _collect_predictions(model, val_loader_final, device)
        prob_matrix = _ensure_probability_matrix(probs)
        frame_labels = _flatten_labels_to_frames(y_val, label_mode=label_mode)
        frame_labels = frame_labels.reshape(frame_labels.shape[0], -1)
        if prob_matrix.shape[0] != frame_labels.shape[0]:
            raise RuntimeError(
                "El número de ejemplos de validación no coincide entre las etiquetas y las predicciones."
            )
        prob_flat = prob_matrix.ravel()
        labels_flat = frame_labels.ravel().astype(int)
        preds_flat = (prob_flat >= 0.5).astype(int)

        fold_results.append(
            FoldResult(
                fold=fold_idx,
                accuracy=float(accuracy_score(labels_flat, preds_flat)),
                precision=float(precision_score(labels_flat, preds_flat, zero_division=0)),
                recall=float(recall_score(labels_flat, preds_flat, zero_division=0)),
                f1=float(f1_score(labels_flat, preds_flat, zero_division=0)),
                average_precision=safe_average_precision(labels_flat, prob_flat),
                roc_auc=safe_roc_auc(labels_flat, prob_flat),
            )
        )

        predictions_df = _window_predictions(
            prob_matrix,
            y_val,
            patients[val_idx],
            records[val_idx],
            label_mode=label_mode,
            fold_idx=fold_idx,
        )
        all_predictions.append(predictions_df)

    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    return fold_results, predictions


def train_full_dataset(
    train_data: DatasetBundle,
    config: PipelineConfig,
    val_data: DatasetBundle | None = None,
    *,
    epoch_time_log_path: Path | None = None,
    epoch_time_log_context: dict[str, object] | None = None,
    save_metric_checkpoints: bool = True,
    checkpoint_dir: Path | None = None,
) -> tuple[nn.Module, dict[str, list[float]], dict[str, float] | None, pd.DataFrame | None, StandardScaler | None]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_sequences = train_data.sequences
    train_features = train_data.features
    train_labels = train_data.labels.astype(np.float32)
    feature_names = list(getattr(train_data, "feature_names", []))
    label_mode = getattr(train_data, "label_mode", "window")
    if config.time_step_labels and label_mode != "time_step":
        raise ValueError("El dataset de entrenamiento no contiene etiquetas por frame requeridas.")
    if not config.time_step_labels and label_mode == "time_step":
        print("⚠️  Advertencia: se detectaron etiquetas por frame, pero el modelo usará salida por ventana.")

    input_shape = train_sequences.shape[1:]
    feat_dim = train_features.shape[1] if train_features is not None else None
    val_label_mode = getattr(val_data, "label_mode", label_mode) if val_data is not None else label_mode

    if val_data is not None and val_data.sequences.size > 0:
        X_train = train_sequences
        y_train = train_labels
        train_features_raw = train_features

        X_val = val_data.sequences
        y_val = val_data.labels.astype(np.float32)
        val_features_raw = val_data.features
        val_patients_info = val_data.patients
        val_records_info = val_data.records
    else:
        X_train = train_sequences
        y_train = train_labels
        train_features_raw = train_features
        X_val = None
        y_val = None
        val_features_raw = None
        val_patients_info = None
        val_records_info = None

        train_patients = train_data.patients
        train_records = train_data.records
        unique_patients = np.unique(train_patients)
        if config.final_validation_split > 0.0 and unique_patients.size >= 2:
            rng = np.random.default_rng(config.seed)
            shuffled_patients = unique_patients.copy()
            rng.shuffle(shuffled_patients)
            val_count = max(1, int(round(unique_patients.size * config.final_validation_split)))
            if val_count >= unique_patients.size:
                val_count = unique_patients.size - 1
            val_patient_ids = set(shuffled_patients[:val_count]) if val_count > 0 else set()
            val_mask = np.isin(train_patients, list(val_patient_ids))
            train_mask = ~val_mask
            if not np.any(train_mask):
                raise RuntimeError(
                    "La división final dejó sin pacientes para entrenamiento. Reduce 'final_validation_split'."
                )
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            X_train = train_sequences[train_idx]
            y_train = train_labels[train_idx]
            train_features_raw = train_features[train_idx] if train_features is not None else None
            if val_idx.size > 0:
                X_val = train_sequences[val_idx]
                y_val = train_labels[val_idx]
                val_features_raw = train_features[val_idx] if train_features is not None else None
                val_patients_info = train_patients[val_idx]
                val_records_info = train_records[val_idx]

    scaler: StandardScaler | None = None
    X_train_feat = None
    X_val_feat = None
    if train_features_raw is not None:
        scaler = StandardScaler()
        X_train_feat = scaler.fit_transform(train_features_raw)
        if val_features_raw is not None:
            X_val_feat = scaler.transform(val_features_raw)
    elif val_features_raw is not None:
        raise RuntimeError(
            "Se proporcionaron features para validación pero no para entrenamiento. Revisa 'include_features'."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(
        model_type=config.model,
        input_shape=input_shape,
        feat_dim=feat_dim,
        num_filters=config.num_filters,
        kernel_size=config.kernel_size,
        dropout_rate=config.dropout,
        rnn_units=config.rnn_units,
        time_step=config.time_step_labels,
    ).to(device)

    optimizer = create_optimizer(config, model)
    scheduler_cfg = create_scheduler_callbacks(
        has_validation=X_val is not None,
        patience=config.patience,
        min_lr=config.min_lr,
        lr_schedule_type=config.lr_schedule_type,
        learning_rate=config.learning_rate,
        cosine_period=config.cosine_annealing_period,
        cosine_min_lr=config.cosine_annealing_min_lr,
    )
    scheduler = scheduler_cfg["scheduler_factory"](optimizer)  # type: ignore[index]
    early_stopping = EarlyStoppingState(config.patience, mode="max")
    loss_computer = create_loss_factory(config)()

    train_dataset = build_torch_dataset_from_arrays(
        (X_train, X_train_feat) if X_train_feat is not None else X_train,
        y_train,
        label_mode=label_mode,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)

    if X_val is not None and y_val is not None:
        val_dataset = build_torch_dataset_from_arrays(
            (X_val, X_val_feat) if X_val_feat is not None else X_val,
            y_val,
            label_mode=val_label_mode,
        )
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    else:
        val_dataset = None
        val_loader = None

    class_weight = _compute_class_weights(y_train, label_mode=label_mode, use_class_weights=config.use_class_weights)

    history: dict[str, list[float]] = defaultdict(list)
    log_interval = None
    if config.verbose:
        try:
            total_batches = len(train_loader)
        except TypeError:
            total_batches = None
        else:
            if total_batches:
                log_interval = max(1, total_batches // 5)
    epoch_context = dict(epoch_time_log_context or {})
    epoch_context.setdefault("phase", "final")
    epoch_timer = EpochTimeLogger(epoch_time_log_path, epoch_context) if epoch_time_log_path else None
    stopper = MaxTimeStopping(config.max_training_minutes * 60.0, verbose=config.verbose) if config.max_training_minutes else None
    if stopper is not None:
        stopper.on_train_begin()

    best_state = None
    best_metric = -float("inf")

    checkpoint_callbacks: list[MetricCheckpoint] | None = None
    if save_metric_checkpoints and checkpoint_dir is not None and val_loader is not None:
        final_dir = checkpoint_dir / "final"
        checkpoint_callbacks = create_metric_checkpoint_callbacks(
            final_dir,
            metrics=("pr_auc", "roc_auc", "recall"),
            has_validation=True,
            prefix="final_",
        )

    for epoch in range(config.epochs):
        if epoch_timer is not None:
            epoch_timer.on_epoch_begin()

        train_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_computer,
            device=device,
            class_weight=class_weight,
            log_interval=log_interval,
            progress_prefix=f"   ↻ Epoch {epoch + 1:03d}/{config.epochs}: ",
        )

        history["loss"].append(train_loss)

        train_eval_loss, train_metrics, _ = _evaluate_model(
            model,
            train_eval_loader,
            device=device,
            label_mode=label_mode,
        )
        train_metrics["loss"] = train_eval_loss
        for metric_name, metric_value in train_metrics.items():
            if metric_name == "loss":
                continue
            history[f"train_{metric_name}"].append(float(metric_value))

        if val_loader is not None:
            val_loss, val_metrics, _ = _evaluate_model(
                model,
                val_loader,
                device=device,
                label_mode=val_label_mode,
            )
            monitor_metric = val_metrics.get("pr_auc", 0.0)
            for metric_name, metric_value in val_metrics.items():
                if metric_name == "loss":
                    history["val_loss"].append(float(metric_value))
                else:
                    history[f"val_{metric_name}"].append(float(metric_value))
        else:
            monitor_metric = -train_loss

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(monitor_metric)
        else:
            scheduler.step()

        epoch_metrics_for_log: dict[str, object] = {"train_loss": train_loss}
        epoch_metrics_for_log.update(
            {
                ("train_eval_loss" if key == "loss" else f"train_{key}"): value
                for key, value in train_metrics.items()
            }
        )
        if val_loader is not None:
            epoch_metrics_for_log.update({f"val_{key}": value for key, value in val_metrics.items()})
        else:
            epoch_metrics_for_log["monitor_metric"] = monitor_metric
        duration_logged = (
            epoch_timer.on_epoch_end(epoch, metrics=epoch_metrics_for_log)
            if epoch_timer is not None
            else None
        )
        if duration_logged is not None:
            history.setdefault("epoch_time_sec", []).append(duration_logged)

        if config.verbose:
            train_summary = (
                f"loss={train_loss:.4f}, "
                f"train_acc={train_metrics.get('accuracy', float('nan')):.4f}, "
                f"train_prec={train_metrics.get('precision', float('nan')):.4f}, "
                f"train_rec={train_metrics.get('recall', float('nan')):.4f}, "
                f"train_auc={train_metrics.get('roc_auc', float('nan')):.4f}"
            )
            if val_loader is not None:
                latest_val_loss = history["val_loss"][-1] if history.get("val_loss") else float("nan")
                val_summary = (
                    f"val_loss={latest_val_loss:.4f}, "
                    f"val_acc={val_metrics.get('accuracy', float('nan')):.4f}, "
                    f"val_prec={val_metrics.get('precision', float('nan')):.4f}, "
                    f"val_rec={val_metrics.get('recall', float('nan')):.4f}, "
                    f"val_auc={val_metrics.get('roc_auc', float('nan')):.4f}, "
                    f"val_pr_auc={val_metrics.get('pr_auc', float('nan')):.4f}"
                )
                print(f"   Epoch {epoch + 1:03d}/{config.epochs}: {train_summary}, {val_summary}")
            else:
                print(f"   Epoch {epoch + 1:03d}/{config.epochs}: {train_summary}")

        improved = early_stopping.update(monitor_metric)
        if improved:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metric = monitor_metric

        if checkpoint_callbacks is not None and val_loader is not None:
            val_metrics_with_loss = dict(val_metrics)
            val_metrics_with_loss.setdefault("loss", history["val_loss"][-1])
            for cb in checkpoint_callbacks:
                cb.update(model, val_metrics_with_loss)

        if early_stopping.should_stop():
            print(f"   ↳ Early stopping activado (best_pr_auc={best_metric:.4f}).")
            break
        if stopper is not None and stopper.should_stop():
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics_summary: dict[str, float] | None = None
    val_predictions_df: pd.DataFrame | None = None

    if val_loader is not None and X_val is not None and y_val is not None:
        val_loss, val_metrics, val_probs = _evaluate_model(
            model,
            val_loader,
            device=device,
            label_mode=val_label_mode,
        )
        val_metrics["loss"] = val_loss
        val_metrics_summary = val_metrics
        val_predictions_df = _window_predictions(
            val_probs,
            y_val,
            val_patients_info if val_patients_info is not None else np.repeat("train_split", y_val.shape[0]),
            val_records_info if val_records_info is not None else np.repeat("train_split", y_val.shape[0]),
            label_mode=val_label_mode,
        )

    return model, history, val_metrics_summary, val_predictions_df, scaler


def save_final_outputs(
    *,
    output_dir: Path,
    config: PipelineConfig,
    model: nn.Module,
    history: dict[str, list[float]],
    val_metrics: dict[str, float] | None,
    eval_metrics: dict[str, float] | None,
    scaler: StandardScaler | None,
    feature_names: list[str],
    val_predictions: pd.DataFrame | None,
    eval_predictions: pd.DataFrame | None,
    dataset_summaries: dict[str, dict[str, object]] | None = None,
    run_id: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model_final.pt"
    torch.save({"state_dict": model.state_dict(), "config": config_to_dict(config)}, model_path)

    history_df = pd.DataFrame(history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_path = output_dir / "training_history.csv"
    history_df.to_csv(history_path, index=False)

    summary = {
        "config": config_to_dict(config),
        "mode": config.mode,
        "history_last": {k: float(v[-1]) for k, v in history.items() if v},
        "validation_metrics": val_metrics,
        "evaluation_metrics": eval_metrics,
        "feature_names": feature_names,
        "val_samples": 0 if val_predictions is None else int(val_predictions.shape[0]),
        "eval_samples": 0 if eval_predictions is None else int(eval_predictions.shape[0]),
        "model_path": str(model_path),
    }
    if config.epoch_time_log_path is not None:
        summary["epoch_time_log"] = str(config.epoch_time_log_path)
    if run_id is not None:
        summary["run_id"] = run_id
    if dataset_summaries:
        summary["dataset_summaries"] = dataset_summaries
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if val_predictions is not None:
        preds_path = output_dir / "final_val_predictions.csv"
        val_predictions.to_csv(preds_path, index=False)

    if eval_predictions is not None:
        eval_path = output_dir / "final_eval_predictions.csv"
        eval_predictions.to_csv(eval_path, index=False)

    if scaler is not None:
        scaler_path = output_dir / "feature_scaler.pkl"
        with scaler_path.open("wb") as fp:
            pickle.dump(scaler, fp)

    print(f"\n📝 Artefactos finales guardados en {output_dir}")


def evaluate_dataset(
    model: nn.Module,
    data: DatasetBundle,
    *,
    batch_size: int,
    scaler: StandardScaler | None,
) -> tuple[dict[str, float], pd.DataFrame]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequences = data.sequences
    labels = data.labels.astype(np.float32)
    features = data.features
    label_mode = getattr(data, "label_mode", "window")

    if features is not None:
        if scaler is None:
            raise RuntimeError("Se requieren features escaladas pero no se encontró scaler entrenado.")
        feat_transformed = scaler.transform(features)
        inputs = (sequences, feat_transformed)
    else:
        inputs = sequences

    dataset = build_torch_dataset_from_arrays(inputs, labels, label_mode=label_mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.eval()

    probs = _collect_predictions(model, dataloader, device)
    metrics = _compute_evaluation_metrics(probs, labels, label_mode=label_mode)
    metrics["loss"] = 0.0

    predictions = _window_predictions(
        probs,
        labels,
        data.patients,
        data.records,
        label_mode=label_mode,
    )

    return metrics, predictions


# =============================================================================
# Punto de entrada basado en configuración
# =============================================================================


def configure_gpu_memory_growth() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def main(argv: list[str] | None = None) -> int:
    config_path = argv[0] if argv else None
    try:
        config = validate_config(load_config(config_path))
    except Exception as err:  # pylint: disable=broad-except
        print(f"✖️  Error al cargar la configuración: {err}", file=sys.stderr)
        return 1

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    configure_gpu_memory_growth()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    base_output_dir = config.output_dir if config.output_dir is not None else (Path.cwd() / "runs")
    base_output_dir = Path(base_output_dir).expanduser().resolve()
    run_output_dir = base_output_dir / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Artefactos de esta corrida en {run_output_dir}")

    config.output_dir = run_output_dir
    if config.checkpoint_dir is not None:
        config.checkpoint_dir = Path(config.checkpoint_dir) / run_id

    if config.dataset_storage in {"memmap", "auto"}:
        if config.dataset_memmap_dir is None:
            config.dataset_memmap_dir = run_output_dir / "memmap"
        config.dataset_memmap_dir.mkdir(parents=True, exist_ok=True)

    if config_path:
        config_source_path = Path(config_path).expanduser()
        if not config_source_path.is_absolute():
            config_source_path = (Path.cwd() / config_source_path).resolve()
    else:
        config_source_path = Path(DEFAULT_CONFIG_FILENAME).expanduser()
        if not config_source_path.is_absolute():
            config_source_path = (Path.cwd() / config_source_path).resolve()

    copied_config_path: Path | None = None
    if config_source_path.exists():
        try:
            copied_config_path = run_output_dir / config_source_path.name
            if config_source_path.resolve() != copied_config_path.resolve():
                copied_config_path.write_text(config_source_path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception as err:  # pylint: disable=broad-except
            print(
                f"⚠️  No se pudo copiar el archivo de configuración original "
                f"({config_source_path}): {err}"
            )

    epoch_time_log_path = resolve_epoch_time_log_path(config, run_id)
    config.epoch_time_log_path = epoch_time_log_path
    epoch_time_context_base = {"run_id": run_id, "mode": config.mode, "model": config.model}
    print(f"🕒 Registrando tiempos de época en {epoch_time_log_path}")
    checkpoint_dir: Path | None = None
    if config.save_metric_checkpoints:
        checkpoint_dir = resolve_checkpoint_dir(config, run_id)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"📦 Checkpoints métricos en {checkpoint_dir}")

    config_dump = config_to_dict(config)
    config_dump["run_id"] = run_id
    config_dump["output_base_dir"] = str(base_output_dir)
    config_dump["config_source"] = str(config_source_path)
    if copied_config_path is not None:
        config_dump["config_copy"] = str(copied_config_path)
    config_dump_path = run_output_dir / "config_used.json"
    config_dump_path.write_text(json.dumps(config_dump, indent=2, sort_keys=True), encoding="utf-8")

    preprocess_settings = resolve_preprocess_settings(config)

    train_dataset: DatasetBundle | None = None
    val_dataset: DatasetBundle | None = None
    eval_dataset: DatasetBundle | None = None
    dataset_summaries: dict[str, dict[str, object]] = {}

    auto_memmap_threshold_bytes: int | None = None
    if (
        config.dataset_storage == "auto"
        and config.dataset_auto_memmap_threshold_mb is not None
        and config.dataset_auto_memmap_threshold_mb > 0
    ):
        auto_memmap_threshold_bytes = int(config.dataset_auto_memmap_threshold_mb * 1024 * 1024)

    def export_cfg_for(split_name: str) -> TFRecordExportConfig | None:
        if config.tfrecord_dir is None:
            return None
        write_enabled = bool(config.write_tfrecords)
        reuse_enabled = bool(config.reuse_existing_tfrecords)
        if not write_enabled and not reuse_enabled:
            return None
        return TFRecordExportConfig(
            base_dir=config.tfrecord_dir,
            split_name=split_name,
            compression=config.tfrecord_compression,
            write_enabled=write_enabled,
            reuse_enabled=reuse_enabled,
        )

    try:
        train_records, _ = collect_records_for_split(config, "train")
        train_dataset = build_windows_dataset(
            train_records,
            condition=config.condition,
            montage=config.montage,
            include_features=config.include_features,
            time_step_labels=config.time_step_labels,
            feature_subset=config.selected_features or None,
            window_sec=config.window_sec,
            hop_sec=config.hop_sec,
            eps=config.epsilon,
            target_fs=config.target_fs,
            preprocess_settings=preprocess_settings,
            include_background_only=config.include_background_only_records,
            sampling_strategy=config.sampling_strategy,
            sampling_seed=config.sampling_seed if config.sampling_seed is not None else config.seed,
            tfrecord_export=export_cfg_for("train"),
            storage_mode=config.dataset_storage,
            memmap_dir=config.dataset_memmap_dir,
            memmap_prefix=f"{run_id}_train",
            auto_memmap_threshold_bytes=auto_memmap_threshold_bytes,
            feature_worker_processes=config.feature_worker_processes,
            feature_worker_chunk_size=config.feature_worker_chunk_size,
            feature_parallel_min_windows=config.feature_parallel_min_windows,
            force_memmap_after_build=config.dataset_force_memmap_after_build,
        )
        dataset_summaries["train"] = summarize_dataset_bundle("train", train_dataset)

        if config.mode == "final":
            val_records, _ = collect_records_for_split(config, "val")
            if val_records:
                val_dataset = build_windows_dataset(
                    val_records,
                    condition=config.condition,
                    montage=config.montage,
                    include_features=config.include_features,
                    time_step_labels=config.time_step_labels,
                    feature_subset=config.selected_features or None,
                    window_sec=config.window_sec,
                    hop_sec=config.hop_sec,
                    eps=config.epsilon,
                    target_fs=config.target_fs,
                    preprocess_settings=preprocess_settings,
                    include_background_only=config.include_background_only_records,
                    sampling_strategy=config.sampling_strategy,
                    sampling_seed=config.sampling_seed if config.sampling_seed is not None else config.seed,
                    tfrecord_export=export_cfg_for("val"),
                    storage_mode=config.dataset_storage,
                    memmap_dir=config.dataset_memmap_dir,
                    memmap_prefix=f"{run_id}_val",
                    auto_memmap_threshold_bytes=auto_memmap_threshold_bytes,
                    feature_worker_processes=config.feature_worker_processes,
                    feature_worker_chunk_size=config.feature_worker_chunk_size,
                    feature_parallel_min_windows=config.feature_parallel_min_windows,
                    force_memmap_after_build=config.dataset_force_memmap_after_build,
                )
                dataset_summaries["val"] = summarize_dataset_bundle("val", val_dataset)

            eval_records, _ = collect_records_for_split(config, "eval")
            if eval_records:
                eval_dataset = build_windows_dataset(
                    eval_records,
                    condition=config.condition,
                    montage=config.montage,
                    include_features=config.include_features,
                    time_step_labels=config.time_step_labels,
                    feature_subset=config.selected_features or None,
                    window_sec=config.window_sec,
                    hop_sec=config.hop_sec,
                    eps=config.epsilon,
                    target_fs=config.target_fs,
                    preprocess_settings=preprocess_settings,
                    include_background_only=config.include_background_only_records,
                    sampling_strategy=config.sampling_strategy,
                    sampling_seed=config.sampling_seed if config.sampling_seed is not None else config.seed,
                    tfrecord_export=export_cfg_for("eval"),
                    storage_mode=config.dataset_storage,
                    memmap_dir=config.dataset_memmap_dir,
                    memmap_prefix=f"{run_id}_eval",
                    auto_memmap_threshold_bytes=auto_memmap_threshold_bytes,
                    feature_worker_processes=config.feature_worker_processes,
                    feature_worker_chunk_size=config.feature_worker_chunk_size,
                    feature_parallel_min_windows=config.feature_parallel_min_windows,
                    force_memmap_after_build=config.dataset_force_memmap_after_build,
                )
                dataset_summaries["eval"] = summarize_dataset_bundle("eval", eval_dataset)
    except Exception as err:  # pylint: disable=broad-except
        traceback.print_exc()
        print(f"✖️  Error al preparar los datasets: {err}", file=sys.stderr)
        return 1

    if config.dry_run:
        print("\n(Dry-run) Fin del proceso. No se entrenó ningún modelo.")
        return 0

    if config.mode == "cv":
        try:
            fold_results, predictions_df = run_group_cv(
                train_dataset,
                model_type=config.model,
                batch_size=config.batch_size,
                epochs=config.epochs,
                num_filters=config.num_filters,
                kernel_size=config.kernel_size,
                dropout_rate=config.dropout,
                rnn_units=config.rnn_units,
                time_step=config.time_step_labels,
                patience=config.patience,
                min_lr=config.min_lr,
                folds=config.folds,
                random_seed=config.seed,
                verbose=config.verbose,
                optimizer_factory=create_optimizer_factory(config),
                optimizer_description=describe_optimizer(config),
                loss_factory=create_loss_factory(config),
                loss_description=describe_loss(config),
                use_class_weights=config.use_class_weights,
                epoch_time_log_path=epoch_time_log_path,
                epoch_time_log_context=epoch_time_context_base,
                lr_schedule_type=config.lr_schedule_type,
                cosine_annealing_period=config.cosine_annealing_period,
                cosine_annealing_min_lr=config.cosine_annealing_min_lr,
                base_learning_rate=config.learning_rate,
                save_metric_checkpoints=config.save_metric_checkpoints,
                checkpoint_dir=checkpoint_dir if config.save_metric_checkpoints else None,
                verbose_level=config.verbose,
                max_training_minutes=config.max_training_minutes,
            )
        except Exception as err:  # pylint: disable=broad-except
            traceback.print_exc()
            print(f"✖️  Error durante el entrenamiento: {err}", file=sys.stderr)
            return 2

        metrics_df = pd.DataFrame([asdict(fr) for fr in fold_results])
        print("\n📈 Métricas por fold:")
        print(metrics_df.round(4).to_string(index=False))
        print("\nResumen promedio:")
        print(metrics_df.drop(columns=["fold"]).mean(numeric_only=True).round(4))

        if config.output_dir:
            save_cv_outputs(
                output_dir=config.output_dir,
                config=config,
                fold_results=fold_results,
                predictions_df=predictions_df,
                feature_names=train_dataset.feature_names,
                run_id=run_id,
            )
    else:
        try:
            model, history, val_metrics, val_predictions, scaler = train_full_dataset(
                train_dataset,
                config,
                val_dataset,
                epoch_time_log_path=epoch_time_log_path,
                epoch_time_log_context=epoch_time_context_base,
                save_metric_checkpoints=config.save_metric_checkpoints,
                checkpoint_dir=checkpoint_dir if config.save_metric_checkpoints else None,
            )
        except Exception as err:  # pylint: disable=broad-except
            traceback.print_exc()
            print(f"✖️  Error durante el entrenamiento final: {err}", file=sys.stderr)
            return 2

        print("\n✅ Entrenamiento final completado.")
        for key, values in history.items():
            if values:
                print(f"   {key}: {values[-1]:.4f}")
        if val_metrics:
            print("\n📏 Métricas del conjunto de validación:")
            for key, value in val_metrics.items():
                print(f"   {key}: {value:.4f}")
        elif config.final_validation_split > 0.0 and (val_dataset is None or val_dataset.sequences.size == 0):
            print("\n⚠️  No se pudo generar un conjunto de validación final (muy pocos pacientes).")

        eval_metrics = None
        eval_predictions = None
        if eval_dataset is not None:
            try:
                eval_metrics, eval_predictions = evaluate_dataset(
                    model,
                    eval_dataset,
                    batch_size=config.batch_size,
                    scaler=scaler,
                )
            except Exception as err:  # pylint: disable=broad-except
                print(f"⚠️  Error al evaluar sobre el split 'eval': {err}")
            else:
                print("\n🎯 Métricas del conjunto de evaluación:")
                for key, value in eval_metrics.items():
                    print(f"   {key}: {value:.4f}")
        else:
            print("\nℹ️  Sin registros en el split 'eval'; se omite la evaluación final.")

        if config.output_dir:
            save_final_outputs(
                output_dir=config.output_dir,
                config=config,
                model=model,
                history=history,
                val_metrics=val_metrics,
                eval_metrics=eval_metrics,
                scaler=scaler,
                feature_names=train_dataset.feature_names,
                val_predictions=val_predictions,
                eval_predictions=eval_predictions,
                dataset_summaries=dataset_summaries,
                run_id=run_id,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
