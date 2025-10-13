#!/usr/bin/env python
"""
Usage example:
    python run_nn_pipeline.py path/al/config.json
If no argumento se pasa, se utilizará ``nn_pipeline_config.json`` en el directorio actual.
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import pickle
import sys
import time
import traceback
import hashlib
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import shutil
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
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

from dataset import DatasetBundle, prepare_model_inputs, resolve_cache_target, _flatten_labels_to_frames, _window_level_labels, _ensure_probability_matrix, build_tf_dataset_from_arrays, collect_records_for_split, build_windows_dataset, summarize_dataset_bundle
from utils import TFRecordExportConfig, PipelineConfig, FoldResult, config_to_dict, load_config, validate_config, resolve_preprocess_settings, resolve_epoch_time_log_path, resolve_checkpoint_dir, DEFAULT_CONFIG_FILENAME, save_cv_outputs


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, datetime):
        return value.isoformat()
    return value

from models.Hybrid import build_hybrid
from models.TCN import build_tcn
from models.Transformer import build_transformer

# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------

class EpochTimeLogger(tf.keras.callbacks.Callback):
    def __init__(
        self,
        log_path: Path,
        context: dict[str, object] | None = None,
        *,
        reset: bool = True,
    ) -> None:
        super().__init__()
        self.log_path = Path(log_path)
        self._base_context = dict(context or {})
        self.context = dict(self._base_context)
        self._fieldnames = ["timestamp", "epoch", "duration_sec", "metrics_json", "context_json"]
        self._start_time: float | None = None
        self._prepare_file(reset)

    def _prepare_file(self, reset: bool) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.log_path.exists():
            try:
                self.log_path.unlink()
            except OSError:
                pass
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            with self.log_path.open("w", encoding="utf-8", newline="") as fp:
                csv.DictWriter(fp, fieldnames=self._fieldnames).writeheader()

    def update_context(self, extra: dict[str, object] | None = None) -> None:
        self.context = dict(self._base_context)
        if extra:
            self.context.update(extra)

    def on_epoch_begin(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        del logs
        self._start_time = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        metrics = dict(logs or {})
        if self._start_time is None:
            return
        duration = time.perf_counter() - self._start_time
        metrics["epoch_time_sec"] = duration
        if logs is not None:
            logs["epoch_time_sec"] = duration
        timestamp = datetime.now(timezone.utc).isoformat()
        metrics_json = json.dumps(metrics, ensure_ascii=False, sort_keys=True, default=_json_default)
        context_json = json.dumps(self.context, ensure_ascii=False, sort_keys=True, default=_json_default)
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


class MaxTimeStopping(tf.keras.callbacks.Callback):
    def __init__(self, max_duration_seconds: float, *, verbose: int = 0) -> None:
        super().__init__()
        if max_duration_seconds <= 0:
            raise ValueError("'max_duration_seconds' debe ser > 0.")
        self.max_duration_seconds = float(max_duration_seconds)
        self.verbose = int(verbose)
        self._start_time: float | None = None
        self._stop_triggered = False

    def on_train_begin(self, logs: dict[str, float] | None = None) -> None:
        del logs
        self._start_time = time.perf_counter()
        self._stop_triggered = False

    def on_batch_end(self, batch: int, logs: dict[str, float] | None = None) -> None:
        del batch, logs
        if self._start_time is None or self._stop_triggered:
            return
        elapsed = time.perf_counter() - self._start_time
        if elapsed >= self.max_duration_seconds:
            self._stop_triggered = True
            if self.verbose:
                minutes = self.max_duration_seconds / 60.0
                print(
                    f"   ↳ MaxTimeStopping detuvo el entrenamiento tras {minutes:.2f} min (límite alcanzado)."
                )
            self.model.stop_training = True

    def on_train_end(self, logs: dict[str, float] | None = None) -> None:
        del logs
        self._start_time = None


class CosineAnnealingScheduler(tf.keras.callbacks.Callback):
    def __init__(
        self,
        *,
        initial_lr: float,
        min_lr: float,
        period: int,
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self.initial_lr = float(initial_lr)
        self.min_lr = float(min_lr)
        self.period = max(1, int(period))
        self.verbose = int(verbose)

    def _compute_lr(self, epoch: int) -> float:
        cycle_epoch = epoch % self.period
        cosine_term = 0.5 * (1.0 + math.cos(math.pi * cycle_epoch / self.period))
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_term

    def on_epoch_begin(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        del logs
        lr_value = self._compute_lr(epoch)
        optimizer = self.model.optimizer
        lr_attr = getattr(optimizer, "learning_rate", None)
        if hasattr(lr_attr, "assign"):
            lr_attr.assign(lr_value)
        elif hasattr(getattr(optimizer, "lr", None), "assign"):
            optimizer.lr.assign(lr_value)  # type: ignore[attr-defined]
        else:
            setattr(optimizer, "learning_rate", lr_value)
        if self.verbose:
            print(f"   ↳ CosineAnnealingScheduler: lr -> {lr_value:.6f}")

    def get_config(self) -> dict[str, float | int]:
        return {
            "initial_lr": self.initial_lr,
            "min_lr": self.min_lr,
            "period": self.period,
            "verbose": self.verbose,
        }

class TverskyLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        *,
        alpha: float,
        beta: float,
        gamma: float = 1.0,
        name: str = "tversky_loss",
    ) -> None:
        super().__init__(name=name)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        y_true_flat = tf.reshape(y_true, (tf.shape(y_true)[0], -1))
        y_pred_flat = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1))

        true_pos = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
        false_pos = tf.reduce_sum((1.0 - y_true_flat) * y_pred_flat, axis=1)
        false_neg = tf.reduce_sum(y_true_flat * (1.0 - y_pred_flat), axis=1)

        denominator = true_pos + self.alpha * false_pos + self.beta * false_neg + epsilon
        tversky_index = (true_pos + epsilon) / denominator
        loss = tf.pow(1.0 - tversky_index, self.gamma)
        return tf.reduce_mean(loss)

    def get_config(self) -> dict[str, float]:
        base = super().get_config()
        base.update({
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        })
        return base


def create_loss_factory(config: PipelineConfig) -> Callable[[], tf.keras.losses.Loss | str]:
    loss_type = config.loss_type

    if loss_type == "binary_crossentropy":
        return lambda: "binary_crossentropy"

    if loss_type == "focal":
        def _factory() -> tf.keras.losses.Loss:
            return tf.keras.losses.BinaryFocalCrossentropy(
                gamma=config.focal_gamma,
                alpha=config.focal_alpha,
                from_logits=False,
                name="binary_focal_crossentropy",
            )

        return _factory

    if loss_type == "tversky":
        def _factory_tversky() -> tf.keras.losses.Loss:
            return TverskyLoss(
                alpha=config.tversky_alpha,
                beta=config.tversky_beta,
                gamma=1.0,
                name="tversky_loss",
            )

        return _factory_tversky

    if loss_type == "tversky_focal":
        def _factory_tversky_focal() -> tf.keras.losses.Loss:
            return TverskyLoss(
                alpha=config.tversky_alpha,
                beta=config.tversky_beta,
                gamma=config.tversky_gamma,
                name="tversky_focal_loss",
            )

        return _factory_tversky_focal

    raise ValueError(f"Tipo de pérdida no soportado: {loss_type}")


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


def create_optimizer_factory(config: PipelineConfig) -> Callable[[], tf.keras.optimizers.Optimizer]:
    def factory() -> tf.keras.optimizers.Optimizer:
        base_kwargs: dict[str, object] = {
            "learning_rate": config.learning_rate,
        }
        if config.optimizer_use_ema:
            base_kwargs.update({
                "use_ema": True,
                "ema_momentum": config.optimizer_ema_momentum,
            })

        if config.optimizer == "adam":
            return tf.keras.optimizers.Adam(**base_kwargs)
        if config.optimizer == "adamw":
            return tf.keras.optimizers.AdamW(
                weight_decay=config.optimizer_weight_decay,
                **base_kwargs,
            )
        raise ValueError(f"Optimizador no soportado: {config.optimizer}")

    return factory


def create_optimizer(config: PipelineConfig) -> tf.keras.optimizers.Optimizer:
    return create_optimizer_factory(config)()


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
    verbose: int,
) -> list[tf.keras.callbacks.Callback]:
    callbacks: list[tf.keras.callbacks.Callback] = []
    monitor_metric = "val_pr_auc" if has_validation else "loss"
    mode = "max" if has_validation else "min"

    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            mode=mode,
            restore_best_weights=True,
        )
    )

    lr_schedule_type = lr_schedule_type.lower()

    if lr_schedule_type == "plateau":
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor_metric,
                factor=0.5,
                patience=max(1, patience // 2),
                min_lr=min_lr,
                mode=mode,
            )
        )
    else:
        target_min_lr = cosine_min_lr if cosine_min_lr is not None else min_lr
        callbacks.append(
            CosineAnnealingScheduler(
                initial_lr=learning_rate,
                min_lr=target_min_lr,
                period=cosine_period,
                verbose=verbose,
            )
        )

    return callbacks


def create_metric_checkpoint_callbacks(
    base_dir: Path,
    *,
    metrics: Sequence[str],
    has_validation: bool,
    prefix: str = "",
    verbose: int = 0,
) -> list[tf.keras.callbacks.Callback]:
    base_dir.mkdir(parents=True, exist_ok=True)
    callbacks: list[tf.keras.callbacks.Callback] = []
    for metric in metrics:
        monitor = f"val_{metric}" if has_validation else metric
        filepath = base_dir / f"{prefix}{metric}_best.weights.h5"
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(filepath),
                monitor=monitor,
                mode="max",
                save_best_only=True,
                save_weights_only=True,
                verbose=verbose,
            )
        )
    return callbacks


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.any(y_true == 1):
        return float(average_precision_score(y_true, y_score))
    return float("nan")


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size > 1:
        return float(roc_auc_score(y_true, y_score))
    return float("nan")


def _transformer_params_from_config(config: PipelineConfig) -> dict[str, object]:
    return {
        "embed_dim": config.transformer_embed_dim,
        "num_layers": config.transformer_num_layers,
        "num_heads": config.transformer_num_heads,
        "mlp_dim": config.transformer_mlp_dim,
        "dropout_rate": config.transformer_dropout,
        "use_se": config.transformer_use_se,
        "se_ratio": config.transformer_se_ratio,
        "use_reconstruction_head": config.transformer_use_reconstruction_head,
        "recon_weight": config.transformer_recon_weight,
        "recon_target": config.transformer_recon_target,
        "koopman_latent_dim": config.transformer_koopman_latent_dim,
        "koopman_loss_weight": config.transformer_koopman_loss_weight,
        "bottleneck_dim": config.transformer_bottleneck_dim,
        "expand_dim": config.transformer_expand_dim,
    }


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
    transformer_params: dict[str, object] | None = None,
) -> tf.keras.Model:
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
            separable=False,
            se_ratio=16,
            feat_input_dim=feat_dim,
            koopman_latent_dim=0,
            koopman_loss_weight=0.0,
            use_reconstruction_head=False,
        )
    if model_type == "hybrid":
        return build_hybrid(
            **common_kwargs,
            num_filters=num_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            time_step=time_step,
            se_ratio=16,
            rnn_units=rnn_units,
            feat_input_dim=feat_dim,
        )
    if model_type == "transformer":
        params = dict(transformer_params or {})
        bottleneck_dim = params.get("bottleneck_dim")
        expand_dim = params.get("expand_dim")
        return build_transformer(
            input_shape=input_shape,
            num_classes=1,
            embed_dim=int(params.get("embed_dim", 128)),
            num_layers=int(params.get("num_layers", 4)),
            num_heads=int(params.get("num_heads", 4)),
            mlp_dim=int(params.get("mlp_dim", 256)),
            dropout_rate=float(params.get("dropout_rate", dropout_rate)),
            time_step_classification=time_step,
            one_hot=False,
            use_se=bool(params.get("use_se", False)),
            se_ratio=int(params.get("se_ratio", 16)),
            feat_input_dim=feat_dim,
            koopman_latent_dim=int(params.get("koopman_latent_dim", 0)),
            koopman_loss_weight=float(params.get("koopman_loss_weight", 0.0)),
            use_reconstruction_head=bool(params.get("use_reconstruction_head", False)),
            recon_weight=float(params.get("recon_weight", 0.0)),
            recon_target=str(params.get("recon_target", "signal")),
            bottleneck_dim=None if bottleneck_dim is None else int(bottleneck_dim),
            expand_dim=None if expand_dim is None else int(expand_dim),
        )
    raise ValueError(f"Modelo no soportado: {model_type}")


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
    use_tf_dataset: bool = False,
    tf_data_shuffle_buffer: int | None = None,
    tf_data_prefetch: int | None = None,
    optimizer_factory: Callable[[], tf.keras.optimizers.Optimizer] | None = None,
    optimizer_description: str | None = None,
    loss_factory: Callable[[], tf.keras.losses.Loss | str] | None = None,
    loss_description: str | None = None,
    use_class_weights: bool = True,
    tf_data_cache: str | bool | None = None,
    jit_compile: bool = False,
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
    transformer_params: dict[str, object] | None = None,
) -> tuple[list[FoldResult], pd.DataFrame]:
    sequences = data.sequences
    features = data.features
    labels = data.labels.astype(np.float32)
    label_mode = getattr(data, "label_mode", "window")
    feature_names = list(getattr(data, "feature_names", []))
    if time_step and label_mode != "time_step":
        raise ValueError("El dataset no contiene etiquetas por frame necesarias para 'time_step'.")
    if not time_step and label_mode == "time_step":
        print("⚠️  Advertencia: etiquetas por frame detectadas pero el modelo entrenará a nivel ventana.")
    patients = data.patients
    records = data.records

    unique_patients = np.unique(patients).size
    n_splits = max(2, min(folds, unique_patients))
    if n_splits < 2:
        raise RuntimeError("Se requieren al menos 2 pacientes distintos para CV agrupada.")

    input_shape = sequences.shape[1:]
    feat_dim = features.shape[1] if features is not None else None

    group_kfold = GroupKFold(n_splits=n_splits)
    labels_for_split = _window_level_labels(labels, label_mode=label_mode)
    fold_results: list[FoldResult] = []

    all_predictions: list[pd.DataFrame] = []

    if optimizer_description:
        print(f"\n⚙️  Optimizador CV: {optimizer_description}")
    if loss_description:
        print(f"   ↳ Pérdida: {loss_description}")

    epoch_timer: EpochTimeLogger | None = None
    if epoch_time_log_path is not None:
        base_context = dict(epoch_time_log_context or {})
        base_context.setdefault("phase", "cv")
        epoch_timer = EpochTimeLogger(epoch_time_log_path, base_context)

    for fold_idx, (train_idx, val_idx) in enumerate(
        group_kfold.split(sequences, labels_for_split, groups=patients), start=1
    ):
        print(
            f"\n🔁 Fold {fold_idx}/{n_splits} | pacientes train={np.unique(patients[train_idx]).size}, "
            f"val={np.unique(patients[val_idx]).size}"
        )

        X_train = sequences[train_idx]
        X_val = sequences[val_idx]
        y_train = labels[train_idx]
        y_val = labels[val_idx]

        if features is not None:
            scaler = StandardScaler()
            X_train_feat = scaler.fit_transform(features[train_idx])
            X_val_feat = scaler.transform(features[val_idx])
        else:
            X_train_feat = X_val_feat = None

        tf.keras.backend.clear_session()
        model = make_model(
            model_type=model_type,
            input_shape=input_shape,
            feat_dim=feat_dim,
            num_filters=num_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            rnn_units=rnn_units,
            time_step=time_step,
            transformer_params=transformer_params,
        )
        optimizer = optimizer_factory() if optimizer_factory else tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss_obj: tf.keras.losses.Loss | str = loss_factory() if loss_factory else "binary_crossentropy"
        model.compile(
            optimizer=optimizer,
            loss=loss_obj,
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            ],
            jit_compile=jit_compile,
        )

        train_frames = _flatten_labels_to_frames(y_train, label_mode=label_mode)
        pos_frac = float(train_frames.mean()) if train_frames.size else 0.0
        class_weight = None
        if use_class_weights:
            if 0 < pos_frac < 1:
                class_weight = {0: 0.5 / (1.0 - pos_frac), 1: 0.5 / pos_frac}
                print(
                    f"   ↳ Balance etiquetas train (frames): {pos_frac:.3f} positivos -> class_weight {class_weight}"
                )
            else:
                print(f"   ↳ Solo una clase en train (pos_frac={pos_frac:.3f}); sin class_weight.")
        else:
            print(f"   ↳ Class weights deshabilitados (pos_frac={pos_frac:.3f}).")

        callbacks = create_scheduler_callbacks(
            has_validation=True,
            patience=patience,
            min_lr=min_lr,
            lr_schedule_type=lr_schedule_type,
            learning_rate=base_learning_rate,
            cosine_period=cosine_annealing_period,
            cosine_min_lr=cosine_annealing_min_lr,
            verbose=verbose_level,
        )

        if max_training_minutes is not None:
            callbacks.append(
                MaxTimeStopping(max_training_minutes * 60.0, verbose=verbose_level)
            )

        if epoch_timer is not None:
            epoch_timer.update_context({"fold": fold_idx})
            callbacks.append(epoch_timer)

        if checkpoint_dir is not None and save_metric_checkpoints:
            fold_dir = checkpoint_dir / f"fold_{fold_idx:02d}"
            callbacks.extend(
                create_metric_checkpoint_callbacks(
                    fold_dir,
                    metrics=("pr_auc", "auc", "precision"),
                    has_validation=True,
                    prefix=f"fold{fold_idx:02d}_",
                    verbose=verbose_level,
                )
            )

        train_inputs_np = prepare_model_inputs(X_train, X_train_feat)
        val_inputs_np = prepare_model_inputs(X_val, X_val_feat)

        fit_kwargs: dict[str, object] = dict(
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
            class_weight=class_weight,
        )

        if use_tf_dataset:
            train_inputs_tf = prepare_model_inputs(X_train, X_train_feat, for_tf_dataset=True)
            val_inputs_tf = prepare_model_inputs(X_val, X_val_feat, for_tf_dataset=True)
            feature_signature = feature_names if feature_names else feat_dim
            cache_train = resolve_cache_target(
                tf_data_cache,
                f"fold{fold_idx}_train",
                label_mode=label_mode,
                feature_signature=feature_signature,
            )
            cache_val = resolve_cache_target(
                tf_data_cache,
                f"fold{fold_idx}_val",
                label_mode=label_mode,
                feature_signature=feature_signature,
            )
            train_dataset = build_tf_dataset_from_arrays(
                train_inputs_tf,
                y_train,
                batch_size=batch_size,
                shuffle=True,
                seed=random_seed,
                shuffle_buffer=tf_data_shuffle_buffer,
                prefetch=tf_data_prefetch,
                cache=cache_train,
            )
            val_dataset = build_tf_dataset_from_arrays(
                val_inputs_tf,
                y_val,
                batch_size=batch_size,
                shuffle=False,
                seed=random_seed,
                shuffle_buffer=None,
                prefetch=tf_data_prefetch,
                cache=cache_val,
            )
            fit_kwargs["x"] = train_dataset
            fit_kwargs["validation_data"] = val_dataset
        else:
            fit_kwargs.update(
                x=train_inputs_np,
                y=y_train,
                batch_size=batch_size,
                validation_data=(val_inputs_np, y_val),
            )

        model.fit(**fit_kwargs)

        val_inputs_for_pred = val_inputs_np if not isinstance(val_inputs_np, list) else val_inputs_np
        raw_probs = model.predict(val_inputs_for_pred, batch_size=batch_size, verbose=0)
        prob_matrix = _ensure_probability_matrix(raw_probs)
        frame_labels = _flatten_labels_to_frames(y_val, label_mode=label_mode)
        if prob_matrix.shape[0] != frame_labels.shape[0]:
            raise RuntimeError(
                "El número de ejemplos de validación no coincide entre las etiquetas y las predicciones."
            )
        prob_matrix = prob_matrix.reshape(prob_matrix.shape[0], -1)
        frame_labels = frame_labels.reshape(frame_labels.shape[0], -1)

        y_val_flat = frame_labels.ravel()
        val_probs_flat = prob_matrix.ravel()
        val_preds_flat = (val_probs_flat >= 0.5).astype(int)
        y_val_int = y_val_flat.astype(int)

        fold_results.append(
            FoldResult(
                fold=fold_idx,
                accuracy=float(accuracy_score(y_val_int, val_preds_flat)),
                precision=float(precision_score(y_val_int, val_preds_flat, zero_division=0)),
                recall=float(recall_score(y_val_int, val_preds_flat, zero_division=0)),
                f1=float(f1_score(y_val_int, val_preds_flat, zero_division=0)),
                average_precision=safe_average_precision(y_val_int, val_probs_flat),
                roc_auc=safe_roc_auc(y_val_int, val_probs_flat),
            )
        )

        window_true = _window_level_labels(y_val, label_mode=label_mode)
        window_prob = prob_matrix.mean(axis=1)
        window_pred = (window_prob >= 0.5).astype(int)

        all_predictions.append(
            pd.DataFrame(
                {
                    "fold": fold_idx,
                    "patient": patients[val_idx],
                    "record": records[val_idx],
                    "y_true": window_true.astype(int),
                    "y_pred": window_pred,
                    "y_prob": window_prob,
                }
            )
        )

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    return fold_results, predictions_df


def train_full_dataset(
    train_data: DatasetBundle,
    config: PipelineConfig,
    val_data: DatasetBundle | None = None,
    *,
    epoch_time_log_path: Path | None = None,
    epoch_time_log_context: dict[str, object] | None = None,
    save_metric_checkpoints: bool = True,
    checkpoint_dir: Path | None = None,
    transformer_params: dict[str, object] | None = None,
) -> tuple[tf.keras.Model, dict[str, list[float]], dict[str, float] | None, pd.DataFrame | None, StandardScaler | None]:
    train_sequences = train_data.sequences
    train_features = train_data.features
    train_labels = train_data.labels.astype(np.float32)
    feature_names = list(getattr(train_data, "feature_names", []))
    label_mode = getattr(train_data, "label_mode", "window")
    if config.time_step_labels and label_mode != "time_step":
        raise ValueError("El dataset de entrenamiento no contiene etiquetas por frame requeridas.")
    if not config.time_step_labels and label_mode == "time_step":
        print("⚠️  Advertencia: se detectaron etiquetas por frame, pero el modelo usará salida por ventana.")
    train_patients = train_data.patients
    train_records = train_data.records

    input_shape = train_sequences.shape[1:]
    feat_dim = train_features.shape[1] if train_features is not None else None
    val_label_mode = getattr(val_data, "label_mode", label_mode) if val_data is not None else label_mode

    # Determine explicit validation data or fallback to patient-based split
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

    tf.keras.backend.clear_session()
    model = make_model(
        model_type=config.model,
        input_shape=input_shape,
        feat_dim=feat_dim,
        num_filters=config.num_filters,
        kernel_size=config.kernel_size,
        dropout_rate=config.dropout,
        rnn_units=config.rnn_units,
        time_step=config.time_step_labels,
        transformer_params=transformer_params,
    )
    optimizer = create_optimizer(config)
    loss_obj = create_loss_factory(config)()
    model.compile(
        optimizer=optimizer,
        loss=loss_obj,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
        jit_compile=config.jit_compile,
    )

    train_frames = _flatten_labels_to_frames(y_train, label_mode=label_mode)
    pos_frac = float(train_frames.mean()) if train_frames.size else 0.0
    class_weight = None
    if config.use_class_weights:
        if 0 < pos_frac < 1:
            class_weight = {0: 0.5 / (1.0 - pos_frac), 1: 0.5 / pos_frac}
            print(
                f"   ↳ Balance etiquetas train (frames): {pos_frac:.3f} positivos -> class_weight {class_weight}"
            )
        else:
            print(f"   ↳ Solo una clase en train (pos_frac={pos_frac:.3f}); sin class_weight.")
    else:
        print(f"   ↳ Class weights deshabilitados (pos_frac={pos_frac:.3f}).")

    print(f"   ↳ Optimizador: {describe_optimizer(config)}")
    print(f"   ↳ Pérdida: {describe_loss(config)}")

    has_validation = X_val is not None and y_val is not None

    callbacks: list[tf.keras.callbacks.Callback] = create_scheduler_callbacks(
        has_validation=has_validation,
        patience=config.patience,
        min_lr=config.min_lr,
        lr_schedule_type=config.lr_schedule_type,
        learning_rate=config.learning_rate,
        cosine_period=config.cosine_annealing_period,
        cosine_min_lr=config.cosine_annealing_min_lr,
        verbose=config.verbose,
    )

    if config.max_training_minutes is not None:
        callbacks.append(
            MaxTimeStopping(config.max_training_minutes * 60.0, verbose=config.verbose)
        )

    epoch_timer: EpochTimeLogger | None = None
    if epoch_time_log_path is not None:
        base_context = dict(epoch_time_log_context or {})
        base_context.setdefault("phase", "final")
        epoch_timer = EpochTimeLogger(epoch_time_log_path, base_context)
        callbacks.append(epoch_timer)

    if checkpoint_dir is not None and save_metric_checkpoints:
        final_dir = checkpoint_dir / "final"
        callbacks.extend(
            create_metric_checkpoint_callbacks(
                final_dir,
                metrics=("pr_auc", "auc", "recall"),
                has_validation=has_validation,
                prefix="final_",
                verbose=config.verbose,
            )
        )

    train_inputs_np = prepare_model_inputs(X_train, X_train_feat)
    fit_kwargs: dict[str, object] = dict(
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=config.verbose,
        class_weight=class_weight,
    )

    if config.use_tf_dataset:
        train_inputs_tf = prepare_model_inputs(X_train, X_train_feat, for_tf_dataset=True)
        feature_signature = feature_names if feature_names else feat_dim
        train_dataset = build_tf_dataset_from_arrays(
            train_inputs_tf,
            y_train,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed,
            shuffle_buffer=config.tf_data_shuffle_buffer,
            prefetch=config.tf_data_prefetch,
            cache=resolve_cache_target(
                config.tf_data_cache,
                "train",
                label_mode=label_mode,
                feature_signature=feature_signature,
            ),
        )
        fit_kwargs["x"] = train_dataset
    else:
        fit_kwargs.update(
            x=train_inputs_np,
            y=y_train,
            batch_size=config.batch_size,
        )

    if X_val is not None and y_val is not None:
        val_inputs_np = prepare_model_inputs(X_val, X_val_feat)
        if config.use_tf_dataset:
            val_inputs_tf = prepare_model_inputs(X_val, X_val_feat, for_tf_dataset=True)
            val_feature_signature = feature_names if feature_names else feat_dim
            val_dataset = build_tf_dataset_from_arrays(
                val_inputs_tf,
                y_val,
                batch_size=config.batch_size,
                shuffle=False,
                seed=config.seed,
                shuffle_buffer=None,
                prefetch=config.tf_data_prefetch,
                cache=resolve_cache_target(
                    config.tf_data_cache,
                    "val",
                    label_mode=val_label_mode,
                    feature_signature=val_feature_signature,
                ),
            )
            fit_kwargs["validation_data"] = val_dataset
        else:
            fit_kwargs["validation_data"] = (val_inputs_np, y_val)

    history = model.fit(**fit_kwargs)

    evaluation: dict[str, float] | None = None
    val_predictions_df: pd.DataFrame | None = None
    if X_val is not None and y_val is not None:
        val_inputs = [X_val, X_val_feat] if X_val_feat is not None else X_val
        eval_results = model.evaluate(
            val_inputs,
            y_val,
            batch_size=config.batch_size,
            verbose=0,
            return_dict=True,
        )
        evaluation = {name: float(value) for name, value in eval_results.items()}
        raw_probs = model.predict(val_inputs, batch_size=config.batch_size, verbose=0)
        prob_matrix = _ensure_probability_matrix(raw_probs)
        prob_matrix = prob_matrix.reshape(prob_matrix.shape[0], -1)
        frame_labels = _flatten_labels_to_frames(y_val, label_mode=val_label_mode)
        frame_labels = frame_labels.reshape(frame_labels.shape[0], -1)
        if prob_matrix.shape[0] != frame_labels.shape[0]:
            raise RuntimeError(
                "El número de ejemplos de validación no coincide entre etiquetas y predicciones."
            )
        y_val_flat = frame_labels.ravel()
        val_probs_flat = prob_matrix.ravel()
        val_preds_flat = (val_probs_flat >= 0.5).astype(int)
        y_val_int = y_val_flat.astype(int)

        evaluation.update(
            {
                "accuracy": float(accuracy_score(y_val_int, val_preds_flat)),
                "precision": float(precision_score(y_val_int, val_preds_flat, zero_division=0)),
                "recall": float(recall_score(y_val_int, val_preds_flat, zero_division=0)),
                "f1": float(f1_score(y_val_int, val_preds_flat, zero_division=0)),
                "pr_auc": safe_average_precision(y_val_int, val_probs_flat),
                "roc_auc": safe_roc_auc(y_val_int, val_probs_flat),
            }
        )

        window_true = _window_level_labels(y_val, label_mode=val_label_mode)
        window_prob = prob_matrix.mean(axis=1)
        window_pred = (window_prob >= 0.5).astype(int)

        if val_patients_info is None or val_records_info is None:
            val_patients_info = np.repeat("train_split", window_true.shape[0])
            val_records_info = np.repeat("train_split", window_true.shape[0])

        val_predictions_df = pd.DataFrame(
            {
                "patient": val_patients_info,
                "record": val_records_info,
                "y_true": window_true.astype(int),
                "y_pred": window_pred,
                "y_prob": window_prob,
            }
        )

    return model, history.history, evaluation, val_predictions_df, scaler


def save_final_outputs(
    *,
    output_dir: Path,
    config: PipelineConfig,
    model: tf.keras.Model,
    history: dict[str, list[float]],
    val_metrics: dict[str, float] | None,
    eval_metrics: dict[str, float] | None,
    scaler: StandardScaler | None,
    feature_names: list[str],
    val_predictions: pd.DataFrame | None,
    eval_predictions: pd.DataFrame | None,
    dataset_summaries: dict[str, dict[str, object]] | None = None,
    run_id: str | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model_final.keras"
    model.save(model_path)

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
    model: tf.keras.Model,
    data: DatasetBundle,
    *,
    batch_size: int,
    scaler: StandardScaler | None,
) -> tuple[dict[str, float], pd.DataFrame]:
    sequences = data.sequences
    labels = data.labels.astype(np.float32)
    features = data.features
    label_mode = getattr(data, "label_mode", "window")

    if features is not None:
        if scaler is None:
            raise RuntimeError("Se requieren features escaladas pero no se encontró scaler entrenado.")
        feat_transformed = scaler.transform(features)
        inputs = [sequences, feat_transformed]
    else:
        inputs = sequences

    eval_results = model.evaluate(inputs, labels, batch_size=batch_size, verbose=0, return_dict=True)
    evaluation = {name: float(value) for name, value in eval_results.items()}

    raw_probs = model.predict(inputs, batch_size=batch_size, verbose=0)
    prob_matrix = _ensure_probability_matrix(raw_probs)
    prob_matrix = prob_matrix.reshape(prob_matrix.shape[0], -1)
    frame_labels = _flatten_labels_to_frames(labels, label_mode=label_mode)
    frame_labels = frame_labels.reshape(frame_labels.shape[0], -1)
    if prob_matrix.shape[0] != frame_labels.shape[0]:
        raise RuntimeError("Las dimensiones de predicción y etiquetas no coinciden durante la evaluación.")

    frame_true = frame_labels.ravel().astype(int)
    frame_probs = prob_matrix.ravel()
    frame_preds = (frame_probs >= 0.5).astype(int)

    evaluation.update(
        {
            "accuracy": float(accuracy_score(frame_true, frame_preds)),
            "precision": float(precision_score(frame_true, frame_preds, zero_division=0)),
            "recall": float(recall_score(frame_true, frame_preds, zero_division=0)),
            "f1": float(f1_score(frame_true, frame_preds, zero_division=0)),
            "pr_auc": safe_average_precision(frame_true, frame_probs),
            "roc_auc": safe_roc_auc(frame_true, frame_probs),
        }
    )

    window_true = _window_level_labels(labels, label_mode=label_mode)
    window_prob = prob_matrix.mean(axis=1)
    window_pred = (window_prob >= 0.5).astype(int)

    predictions = pd.DataFrame(
        {
            "patient": data.patients,
            "record": data.records,
            "y_true": window_true.astype(int),
            "y_pred": window_pred,
            "y_prob": window_prob,
        }
    )

    return evaluation, predictions


# -----------------------------------------------------------------------------
# Config-driven entry point
# -----------------------------------------------------------------------------


def configure_gpu_memory_growth():
    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        return
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

def main(argv: list[str] | None = None) -> int:
    config_path = argv[0] if argv else None
    try:
        config = validate_config(load_config(config_path))
    except Exception as err:  # pylint: disable=broad-except
        print(f"✖️  Error al cargar la configuración: {err}", file=sys.stderr)
        return 1

    np.random.seed(config.seed)
    tf.keras.utils.set_random_seed(config.seed)
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
                shutil.copy2(config_source_path, copied_config_path)
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
            format=config.dataset_cache_format,
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

    transformer_params = _transformer_params_from_config(config) if config.model == "transformer" else None

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
                use_tf_dataset=config.use_tf_dataset,
                tf_data_shuffle_buffer=config.tf_data_shuffle_buffer,
                tf_data_prefetch=config.tf_data_prefetch,
                optimizer_factory=create_optimizer_factory(config),
                optimizer_description=describe_optimizer(config),
                loss_factory=create_loss_factory(config),
                loss_description=describe_loss(config),
                use_class_weights=config.use_class_weights,
                tf_data_cache=config.tf_data_cache,
                jit_compile=config.jit_compile,
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
                transformer_params=transformer_params,
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
                transformer_params=transformer_params,
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
