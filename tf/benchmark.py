#!/usr/bin/env python
"""Benchmark de métricas sobre el split de evaluación (versión TensorFlow).

Este script replica la funcionalidad de ``pt.benchmark_eval`` pero
utilizando el pipeline de TensorFlow. Carga una corrida entrenada, genera (o
reutiliza) las predicciones del split ``eval`` y produce las mismas métricas,
curvas y reportes en el subdirectorio ``benchmark_eval`` del run.

Uso típico::

    python -m tf.benchmark_eval run_dir=runs/mi_experimento/20251005-123456
    python -m tf.benchmark_eval run_dir=runs/mi_experimento/20251005-123456 weights_path=/ruta/pesos.keras

Todos los parámetros se pasan como pares ``clave=valor``. Se soportan tanto
modelos serializados completos (``model_path``) como archivos de pesos
(``weights_path``). Cuando se proporcionan pesos se reconstruye el modelo a
partir de la configuración guardada.
"""
from __future__ import annotations

import contextlib
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from tensorflow.keras import Model

try:
    from tensorflow import keras as _keras  # type: ignore
except Exception:  # pragma: no cover - fallback for standalone Keras installs
    try:
        import keras as _keras  # type: ignore
    except Exception:  # pragma: no cover - keep graceful degradation
        _keras = None  # type: ignore

from dataset import (
    build_tf_dataset_from_arrays,
    build_windows_dataset,
    collect_records_for_split,
    DatasetBundle,
)
from pipeline import (
    _flatten_labels_to_frames,
    _window_level_labels,
    create_loss_factory,
    create_optimizer,
    make_model,
)
from utils import PipelineConfig, load_config, resolve_preprocess_settings, validate_config


@dataclass
class BenchmarkOptions:
    run_dir: Path
    device: str | None = None
    threshold_start: float = 0.05
    threshold_stop: float = 0.95
    threshold_step: float = 0.05
    force_recompute: bool = False
    batch_size: int | None = None
    model_path: Path | None = None
    weights_path: Path | None = None
    best_metric: str = "pr_auc"


def _usage() -> str:
    return (
        "Uso: python -m tf.benchmark_eval run_dir=/ruta/al/run "
        "[device=cpu] [threshold_start=0.05] [threshold_stop=0.95] [threshold_step=0.05] "
        "[force_recompute=true] [batch_size=128] [model_path=/otra/ruta/model.keras] "
        "[weights_path=/otra/ruta/weights.keras] [best_metric=pr_auc]"
    )


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "si", "sí", "on"}


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _parse_cli_options(argv: list[str]) -> BenchmarkOptions:
    if not argv:
        raise SystemExit(_usage())

    options: dict[str, str] = {}
    run_dir: Path | None = None

    for arg in argv:
        token = arg.strip()
        if not token:
            continue
        if token in {"-h", "--help", "help"}:
            raise SystemExit(_usage())
        if "=" not in token:
            if run_dir is None:
                run_dir = Path(token).expanduser().resolve()
                continue
            raise SystemExit(f"Argumento '{token}' inválido. {_usage()}")
        key, value = token.split("=", 1)
        key = key.strip().lower().replace("-", "_")
        options[key] = value.strip()

    if run_dir is None:
        run_dir_value = options.pop("run_dir", None)
        if run_dir_value is None:
            raise SystemExit(_usage())
        run_dir = Path(run_dir_value).expanduser().resolve()
    else:
        options.pop("run_dir", None)

    device = options.pop("device", None)
    threshold_start = float(options.pop("threshold_start", 0.05))
    threshold_stop = float(options.pop("threshold_stop", 0.95))
    threshold_step = float(options.pop("threshold_step", 0.05))
    force_recompute = _parse_bool(options.pop("force_recompute", "false"))
    batch_size = _parse_optional_int(options.pop("batch_size", None))
    model_path_value = options.pop("model_path", None)
    weights_path_value = options.pop("weights_path", None)
    best_metric_raw = options.pop("best_metric", "pr_auc")

    if options:
        unknown = ", ".join(sorted(options.keys()))
        raise SystemExit(f"Argumentos desconocidos: {unknown}\n{_usage()}")

    model_path = Path(model_path_value).expanduser().resolve() if model_path_value else None
    weights_path = Path(weights_path_value).expanduser().resolve() if weights_path_value else None

    if model_path and weights_path:
        raise SystemExit("Proporciona solo 'model_path' o 'weights_path', no ambos a la vez.")

    best_metric = best_metric_raw.strip().lower().replace(" ", "_").replace("-", "_") or "pr_auc"

    return BenchmarkOptions(
        run_dir=run_dir,
        device=device,
        threshold_start=threshold_start,
        threshold_stop=threshold_stop,
        threshold_step=threshold_step,
        force_recompute=force_recompute,
        batch_size=batch_size,
        model_path=model_path,
        weights_path=weights_path,
        best_metric=best_metric,
    )


def _load_run_config(run_dir: Path) -> PipelineConfig:
    config_path = run_dir / "config_used.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró '{config_path}'.")
    config = load_config(config_path)
    config = validate_config(config)
    config.output_dir = run_dir
    if config.dataset_memmap_dir is None and config.dataset_storage in {"memmap", "auto"}:
        config.dataset_memmap_dir = run_dir / "memmap"
    return config


def _ensure_model_compiled(model_obj: Model, cfg: PipelineConfig) -> None:
    if getattr(model_obj, "optimizer", None) is not None and model_obj.compiled_loss is not None:
        return
    optimizer = create_optimizer(cfg)
    loss_fn = create_loss_factory(cfg)()
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
    ]
    model_obj.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics,
        jit_compile=getattr(cfg, "jit_compile", False),
    )


def _clone_model_to_device(model_obj: Model, target_device: str, cfg: PipelineConfig) -> Model:
    device_key = target_device.upper()
    if device_key.startswith("CPU"):
        device_scope = "/CPU:0"
    elif device_key.startswith("GPU"):
        device_scope = "/GPU:0"
    elif device_key.startswith("TPU"):
        device_scope = "/TPU:0"
    else:
        device_scope = "/CPU:0"

    weights = model_obj.get_weights()
    with tf.device(device_scope):
        cloned = tf.keras.models.clone_model(model_obj)
        cloned.set_weights(weights)
    _ensure_model_compiled(cloned, cfg)
    return cloned


def _ensure_device(device_arg: str | None) -> str:
    if not device_arg:
        return "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    return device_arg.upper()


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


def _threshold_sequence(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("'threshold_step' debe ser > 0.")
    seq = np.arange(start, stop + step / 2.0, step, dtype=np.float32)
    seq = np.clip(seq, 0.0, 1.0)
    seq = np.unique(seq)
    return seq


def _compute_false_alarm_rate(
    predictions: pd.DataFrame,
    threshold: float,
    *,
    hop_sec: float,
    window_sec: float,
) -> tuple[int, float]:
    if predictions.empty:
        return 0, float("nan")
    false_alarm_events = 0
    total_hours = 0.0
    for _, group in predictions.groupby("record", sort=False):
        y_true = group["y_true"].to_numpy(dtype=np.int8)
        y_prob = group["y_prob"].to_numpy(dtype=np.float32)
        y_pred = (y_prob >= threshold).astype(np.int8)
        fp_flags = (y_pred == 1) & (y_true == 0)
        if fp_flags.size:
            binary = fp_flags.astype(np.int8)
            transitions = np.diff(np.concatenate(([0], binary)))
            false_alarm_events += int(np.count_nonzero(transitions == 1))
        n_windows = group.shape[0]
        duration_sec = window_sec + max(0, n_windows - 1) * hop_sec
        total_hours += duration_sec / 3600.0
    false_alarms_per_hour = (
        false_alarm_events / total_hours if total_hours > 0 else float("inf")
    )
    return false_alarm_events, false_alarms_per_hour


def _confusion_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    *,
    hop_sec: float,
    window_sec: float,
    predictions_df: pd.DataFrame,
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_alarm_events, false_alarm_rate = _compute_false_alarm_rate(
        predictions_df,
        threshold,
        hop_sec=hop_sec,
        window_sec=window_sec,
    )
    return {
        "threshold": float(threshold),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "fpr": float(fpr),
        "false_alarm_events": float(false_alarm_events),
        "false_alarms_per_hour": float(false_alarm_rate),
    }


def _candidate_checkpoint_dirs(config: PipelineConfig, run_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        expanded = path.expanduser()
        key = str(expanded)
        if key not in seen:
            seen.add(key)
            candidates.append(expanded)

    raw_dir = getattr(config, "checkpoint_dir", None)
    if raw_dir:
        base = Path(raw_dir)
        _add(base)
        if not base.is_absolute():
            anchors = [run_dir, run_dir.parent, run_dir.parent.parent, Path.cwd()]
            for anchor in anchors:
                if anchor:
                    _add(anchor / base)

    default_bases = [
        run_dir / "checkpoints",
        run_dir,
    ]
    # Include siblings like runs/checkpoints/<run_id>
    if run_dir.parent != run_dir:
        default_bases.extend(
            [
                run_dir.parent / "checkpoints",
                run_dir.parent / "checkpoints" / run_dir.name,
            ]
        )
    if run_dir.parent.parent != run_dir.parent:
        default_bases.extend(
            [
                run_dir.parent.parent / "checkpoints",
                run_dir.parent.parent / "checkpoints" / run_dir.name,
            ]
        )

    for path in default_bases:
        _add(path)

    return candidates


def _resolve_best_checkpoint(directories: Iterable[Path], metric: str) -> Path | None:
    sanitized_metric = metric.strip().lower().replace(" ", "_").replace("-", "_")
    name_variants = [
        f"final_{sanitized_metric}_best.weights.h5",
        f"final_{sanitized_metric}_best.h5",
        f"final_{sanitized_metric}_best.keras",
        f"{sanitized_metric}_best.weights.h5",
        f"{sanitized_metric}_best.h5",
        f"{sanitized_metric}_best.keras",
        f"{sanitized_metric}.weights.h5",
        f"{sanitized_metric}.keras",
    ]

    visited: set[Path] = set()
    for base in directories:
        for target in (base, base / "final"):
            if target in visited or not target.exists():
                visited.add(target)
                continue
            visited.add(target)
            for variant in name_variants:
                candidate = target / variant
                if candidate.exists():
                    return candidate

    pattern_variants = [
        f"*{sanitized_metric}*best*.weights.h5",
        f"*{sanitized_metric}*best*.h5",
        f"*{sanitized_metric}*best*.keras",
    ]
    for base in directories:
        if not base.exists():
            continue
        for pattern in pattern_variants:
            matches = sorted(
                base.rglob(pattern),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if matches:
                return matches[0]
    return None


def _infer_metric_from_checkpoint(path: Path) -> str | None:
    name = path.name
    suffixes = (".weights.h5", ".h5", ".keras")
    for suffix in suffixes:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name.lower().endswith("_best"):
        name = name[: -len("_best")]
    parts = [part for part in name.split("_") if part]
    while parts and (parts[0].lower() in {"final"} or parts[0].lower().startswith("fold")):
        parts.pop(0)
    if not parts:
        return None
    metric = "_".join(parts).strip("_")
    return metric or None


def _discover_latest_checkpoint(directories: Iterable[Path]) -> Path | None:
    best_path: Path | None = None
    best_key: tuple[int, float] | None = None
    seen: set[str] = set()
    patterns = ("*.weights.h5", "*.h5", "*.keras")

    for base in directories:
        if not base.exists():
            continue
        for pattern in patterns:
            for candidate in base.rglob(pattern):
                if not candidate.is_file():
                    continue
                key_str = str(candidate.resolve())
                if key_str in seen:
                    continue
                seen.add(key_str)
                name = candidate.name.lower()
                is_best = 0 if "_best" in name else 1
                mtime = -candidate.stat().st_mtime
                key = (is_best, mtime)
                if best_key is None or key < best_key:
                    best_key = key
                    best_path = candidate
    return best_path


def _find_feature_scaler(directories: Iterable[Path]) -> Path | None:
    for base in directories:
        if not base.exists():
            continue
        candidate = base / "feature_scaler.pkl"
        if candidate.exists():
            return candidate
    for base in directories:
        if not base.exists():
            continue
        for candidate in base.rglob("feature_scaler.pkl"):
            if candidate.is_file():
                return candidate
    return None


def _plot_confusion_matrix(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    threshold: float,
    output_path: Path,
) -> None:
    matrix = np.array([[tn, fp], [fn, tp]], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred BG", "Pred Seiz"])
    ax.set_yticks([0, 1], labels=["Real BG", "Real Seiz"])
    ax.set_title(f"Matriz de confusión (threshold={threshold:.2f})")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{int(value)}", ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_curve(
    x: Iterable[float],
    y: Iterable[float],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
    highlight_point: Optional[Tuple[float, float]] = None,
    highlight_label: str | None = None,
    highlight_threshold: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(x, y, label=title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    if xlabel.lower().startswith("fpr"):
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
    if highlight_point is not None:
        hx, hy = highlight_point
        ax.scatter([hx], [hy], color="tab:red", marker="o", s=60, zorder=10)
        label = highlight_label or "Best threshold"
        annotation = label
        if highlight_threshold is not None and not np.isnan(highlight_threshold):
            annotation = f"{label}\nτ={highlight_threshold:.3f}"
        ax.annotate(
            annotation,
            (hx, hy),
            textcoords="offset points",
            xytext=(8, -12),
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, lw=0.5),
        )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_threshold_sweep(thresholds: np.ndarray, f1: np.ndarray, false_alarm_rate: np.ndarray, output_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax1.plot(thresholds, f1, color="tab:blue", marker="o", label="F1")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("F1", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(thresholds, false_alarm_rate, color="tab:red", marker="s", label="FA/h")
    ax2.set_ylabel("Falsas alarmas por hora", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    ax1.set_title("Sweep de thresholds")
    ax1.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_training_curves(history_df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    created: dict[str, Path] = {}
    if history_df.empty:
        return created

    history = history_df.copy()
    if "epoch" not in history.columns:
        history.insert(0, "epoch", np.arange(1, len(history) + 1))

    epochs = history["epoch"].to_numpy()

    train_loss_col: str | None = None
    for candidate in ("loss", "train_loss", "training_loss"):
        if candidate in history.columns:
            train_loss_col = candidate
            break
    val_loss_col: str | None = None
    for candidate in ("val_loss", "validation_loss"):
        if candidate in history.columns:
            val_loss_col = candidate
            break

    if train_loss_col or val_loss_col:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        if train_loss_col:
            ax.plot(epochs, history[train_loss_col], label="Train loss", color="tab:blue")
        if val_loss_col:
            ax.plot(epochs, history[val_loss_col], label="Val loss", color="tab:orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss por época")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        path = output_dir / "history_loss.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        created["loss_plot"] = path

    train_acc_cols = [
        col
        for col in history.columns
        if not col.lower().startswith("val") and col.lower().endswith("accuracy")
    ]
    val_acc_cols = [
        col
        for col in history.columns
        if col.lower().startswith("val") and col.lower().endswith("accuracy")
    ]
    train_acc_col = train_acc_cols[0] if train_acc_cols else None
    val_acc_col = val_acc_cols[0] if val_acc_cols else None
    if not train_acc_col and "accuracy" in history.columns:
        train_acc_col = "accuracy"
    if not val_acc_col and "val_accuracy" in history.columns:
        val_acc_col = "val_accuracy"

    if train_acc_col or val_acc_col:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        if train_acc_col:
            ax.plot(epochs, history[train_acc_col], label="Train accuracy", color="tab:green")
        if val_acc_col:
            ax.plot(epochs, history[val_acc_col], label="Val accuracy", color="tab:red")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Accuracy por época")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        path = output_dir / "history_accuracy.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        created["accuracy_plot"] = path

    plot_specs: list[tuple[str, str | None, str | None, str, str, tuple[float, float] | None]] = []
    if train_loss_col or val_loss_col:
        plot_specs.append(
            (
                "Loss",
                train_loss_col,
                val_loss_col,
                "Loss",
                "Epoch",
                None,
            )
        )
    if train_acc_col or val_acc_col:
        plot_specs.append(
            (
                "Accuracy",
                train_acc_col,
                val_acc_col,
                "Accuracy",
                "Epoch",
                (0.0, 1.0),
            )
        )

    if plot_specs:
        n_subplots = len(plot_specs)
        fig, axes = plt.subplots(1, n_subplots, figsize=(6.0 * n_subplots, 4.0), squeeze=False)
        for idx, (title, train_col, val_col, ylabel, xlabel, ylim) in enumerate(plot_specs):
            ax = axes[0][idx]
            if train_col:
                ax.plot(epochs, history[train_col], label="Train", color="tab:blue")
            if val_col:
                ax.plot(epochs, history[val_col], label="Validation", color="tab:orange")
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(True, linestyle="--", alpha=0.4)
            if train_col and val_col:
                ax.legend()
        fig.tight_layout()
        combined_path = output_dir / "history_training_curves.png"
        fig.savefig(combined_path, dpi=200)
        plt.close(fig)
        created["combined_plot"] = combined_path

    return created


def _prepare_inputs(
    dataset: DatasetBundle,
    *,
    scaler: object | None,
    batch_size: int,
) -> tuple[tf.data.Dataset, np.ndarray]:
    sequences = dataset.sequences
    labels = dataset.labels.astype(np.float32)
    features = dataset.features

    inputs: object
    if features is not None:
        if scaler is None:
            print(
                "⚠️  No se encontró scaler entrenado; se recalculará a partir de las features del split eval."
            )
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            std = np.where(std < 1e-8, 1.0, std)
            feat_transformed = (features - mean) / std
        else:
            feat_transformed = scaler.transform(features)
        inputs = (sequences, feat_transformed)
    else:
        inputs = sequences

    tf_dataset = build_tf_dataset_from_arrays(
        inputs,
        labels,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        shuffle_buffer=None,
        cache=False,
        prefetch=None,
    )
    return tf_dataset, labels


def _prepare_predictions_dataframe(
    model: Model,
    dataset: DatasetBundle,
    tf_dataset: tf.data.Dataset,
    labels: np.ndarray,
    *,
    batch_size: int,
) -> pd.DataFrame:
    raw_probs = model.predict(tf_dataset, verbose=0)
    prob_matrix = np.asarray(raw_probs, dtype=np.float32)
    prob_matrix = prob_matrix.reshape(prob_matrix.shape[0], -1)
    label_mode = getattr(dataset, "label_mode", "window")

    frame_labels = _flatten_labels_to_frames(labels, label_mode=label_mode)
    frame_labels = frame_labels.reshape(frame_labels.shape[0], -1)
    if prob_matrix.shape[0] != frame_labels.shape[0]:
        raise RuntimeError("Las dimensiones de predicción y etiquetas no coinciden durante la evaluación.")

    window_true = _window_level_labels(labels, label_mode=label_mode).astype(np.int8)
    window_prob = prob_matrix.mean(axis=1)

    predictions = pd.DataFrame(
        {
            "patient": dataset.patients,
            "record": dataset.records,
            "y_true": window_true,
            "y_prob": window_prob,
        }
    )
    return predictions


def _run_model_evaluation_once(
    model: Model,
    eval_bundle: DatasetBundle,
    *,
    scaler: object | None,
    batch_size: int,
    device_override: str | None,
) -> tuple[dict[str, float], pd.DataFrame]:
    context = tf.device(device_override) if device_override else contextlib.nullcontext()
    with context:
        tf_dataset, labels = _prepare_inputs(
            eval_bundle,
            scaler=scaler,
            batch_size=batch_size,
        )
        metrics = model.evaluate(tf_dataset, verbose=0, return_dict=True)
        metrics = {name: float(value) for name, value in metrics.items()}
        predictions = _prepare_predictions_dataframe(
            model,
            eval_bundle,
            tf_dataset,
            labels,
            batch_size=batch_size,
        )
    return metrics, predictions


def _evaluate_with_memory_fallback(
    model: Model,
    eval_bundle: DatasetBundle,
    *,
    scaler: object | None,
    initial_batch_size: int,
    prefer_device: str,
    get_cpu_model: Callable[[], Model] | None = None,
) -> tuple[dict[str, float], pd.DataFrame, int, bool]:
    batch_size = max(1, int(initial_batch_size))
    device_override: str | None = None
    used_cpu = False
    oom_errors = (tf.errors.ResourceExhaustedError, tf.errors.InternalError)

    while True:
        try:
            metrics, predictions = _run_model_evaluation_once(
                model,
                eval_bundle,
                scaler=scaler,
                batch_size=batch_size,
                device_override=device_override,
            )
            return metrics, predictions, batch_size, used_cpu
        except oom_errors as exc:
            if device_override == "/CPU:0" or (prefer_device.upper() == "CPU" and batch_size <= 1):
                raise

            upper_message = str(exc).upper()
            if batch_size > 1 and (
                "OOM" in upper_message
                or "RESOURCE_EXHAUSTED" in upper_message
                or "OUT OF MEMORY" in upper_message
            ):
                next_batch = max(1, batch_size // 2)
                if next_batch == batch_size and batch_size > 1:
                    next_batch = batch_size - 1
                if next_batch < batch_size:
                    print(
                        f"⚠️  Memoria agotada durante la evaluación (batch={batch_size}). Reintentando con batch={next_batch}."
                    )
                    batch_size = next_batch
                    continue

            if used_cpu:
                raise

            if get_cpu_model is None:
                raise RuntimeError(
                    "Memoria de GPU agotada durante la evaluación y no se pudo recargar el modelo en CPU."
                ) from exc

            print("⚠️  Memoria de GPU insuficiente; reintentando evaluación en CPU.")
            try:
                model = get_cpu_model()
            except Exception as reload_error:  # pragma: no cover - delegated to caller
                raise RuntimeError(
                    "No se pudo preparar el modelo en CPU durante el fallback de evaluación."
                ) from reload_error
            device_override = "/CPU:0"
            used_cpu = True
            continue
        except tf.errors.InvalidArgumentError as exc:
            if device_override == "/CPU:0" and get_cpu_model is not None:
                message_upper = str(exc).upper()
                if "TRYING TO ACCESS RESOURCE" in message_upper or "LOCATED IN DEVICE" in message_upper:
                    print(
                        "⚠️  El modelo estaba fijado en GPU; se recargará en CPU para completar la evaluación."
                    )
                    try:
                        model = get_cpu_model()
                    except Exception as reload_error:  # pragma: no cover - delegated to caller
                        raise RuntimeError(
                            "Falló la preparación del modelo en CPU tras detectar recursos anclados a GPU."
                        ) from reload_error
                    device_override = "/CPU:0"
                    used_cpu = True
                    continue
            raise


def _load_model_from_sources(
    *,
    base_config: PipelineConfig,
    eval_bundle: DatasetBundle,
    default_model_path: Path,
    model_path: Path | None,
    weights_path: Path | None,
    output_dir: Path,
    device: str,
) -> tuple[Model, PipelineConfig, Path]:
    model_config = base_config

    strategy: tf.distribute.Strategy | None = None
    if device.startswith("TPU"):
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=device)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    elif device == "GPU" and tf.config.list_physical_devices("GPU"):
        strategy = tf.distribute.MirroredStrategy()

    if weights_path is not None and model_path is not None:
        raise SystemExit("Proporciona solo 'model_path' o 'weights_path', no ambos.")

    def _maybe_in_strategy(fn):
        if strategy is None:
            return fn()
        with strategy.scope():
            return fn()

    resolved_weights_path = weights_path
    resolved_model_path = model_path

    if resolved_weights_path is not None:
        suffix_lower = resolved_weights_path.suffix.lower()
        is_weights_file = resolved_weights_path.name.lower().endswith(".weights.h5")
        if resolved_weights_path.is_dir() or (suffix_lower in {".keras", ".h5"} and not is_weights_file):
            # Treat SavedModel directories or .keras full-model artifacts as serialized models.
            resolved_model_path = resolved_weights_path
            resolved_weights_path = None

    if resolved_weights_path is not None:
        if not resolved_weights_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de pesos: {resolved_weights_path}")
        transformer_params = (
            _transformer_params_from_config(model_config)
            if model_config.model == "transformer"
            else None
        )
        # Al cargar solo pesos necesitamos reconstruir el modelo.
        builder = lambda: make_model(
            model_type=model_config.model,
            input_shape=eval_bundle.sequences.shape[1:],
            feat_dim=(eval_bundle.features.shape[1] if eval_bundle.features is not None else None),
            num_filters=model_config.num_filters,
            kernel_size=model_config.kernel_size,
            dropout_rate=model_config.dropout,
            rnn_units=model_config.rnn_units,
            time_step=model_config.time_step_labels,
            transformer_params=transformer_params,
            use_input_se_block=getattr(model_config, "use_input_se_block", False),
            input_se_ratio=getattr(model_config, "input_se_ratio", 8),
            use_input_conv_block=getattr(model_config, "use_input_conv_block", False),
            input_conv_filters=getattr(model_config, "input_conv_filters", 32),
            input_conv_kernel_size=getattr(model_config, "input_conv_kernel_size", 5),
            input_conv_layers=getattr(model_config, "input_conv_layers", 0),
            feature_enricher_units=getattr(model_config, "feature_enricher_units", []),
            feature_enricher_activation=getattr(model_config, "feature_enricher_activation", "relu"),
            feature_enricher_dropout=getattr(model_config, "feature_enricher_dropout", 0.0),
        )
        model = _maybe_in_strategy(builder)
        try:
            model.load_weights(resolved_weights_path)
        except Exception as exc:
            raise RuntimeError(
                "No se pudieron cargar los pesos en la arquitectura reconstruida. "
                "Verifica que la configuración del modelo coincida con la usada durante el entrenamiento "
                "o utiliza un archivo .keras con el modelo serializado completo."
            ) from exc
        _maybe_in_strategy(lambda: _ensure_model_compiled(model, model_config))
        return model, model_config, resolved_weights_path

    resolved_model_path = resolved_model_path or default_model_path
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {resolved_model_path}")

    if resolved_model_path.is_dir():
        model = _maybe_in_strategy(lambda: tf.keras.models.load_model(resolved_model_path))
        config_candidate = getattr(model, "_pipeline_config", None)
        if config_candidate is not None:
            model_config = validate_config(config_candidate)
            model_config.output_dir = base_config.output_dir
        _maybe_in_strategy(lambda: _ensure_model_compiled(model, model_config))
        return model, model_config, resolved_model_path

    if resolved_model_path.suffix in {".h5", ".keras"}:
        custom_objects: dict[str, object] = {}
        try:
            from tf.models.Transformer import (
                FiLM1D,
                MultiHeadSelfAttentionRoPE,
                AttentionPooling1D,
                AddCLSToken,
                gelu,
                rotary_embedding,
            )

            custom_objects.update(
                {
                    "FiLM1D": FiLM1D,
                    "MultiHeadSelfAttentionRoPE": MultiHeadSelfAttentionRoPE,
                    "AttentionPooling1D": AttentionPooling1D,
                    "AddCLSToken": AddCLSToken,
                    "gelu": gelu,
                    "rotary_embedding": rotary_embedding,
                }
            )
        except Exception:
            pass

        try:
            from tf.models.Hybrid import FiLM1D as FiLM1D_h, AttentionPooling1D as AttentionPooling1D_h, gelu as gelu_h

            if "FiLM1D" not in custom_objects:
                custom_objects["FiLM1D"] = FiLM1D_h
            if "AttentionPooling1D" not in custom_objects:
                custom_objects["AttentionPooling1D"] = AttentionPooling1D_h
            if "gelu" not in custom_objects:
                custom_objects["gelu"] = gelu_h
        except Exception:
            pass

        load_kwargs: dict[str, object] = {"custom_objects": custom_objects, "compile": False}

        try:
            model = _maybe_in_strategy(lambda: tf.keras.models.load_model(resolved_model_path, **load_kwargs))
        except ValueError as exc:
            message = str(exc).lower()
            if "lambda" in message:
                print(
                    "⚠️  Modelo contiene capas Lambda con funciones Python; se intentará cargar con deserialización insegura."
                )
                unsafe_kwargs = dict(load_kwargs)
                # Prefer explicit safe_mode when supported.
                try:
                    model = _maybe_in_strategy(
                        lambda: tf.keras.models.load_model(
                            resolved_model_path,
                            safe_mode=False,
                            **unsafe_kwargs,
                        )
                    )
                except TypeError:
                    if _keras is not None:
                        try:
                            _keras.config.enable_unsafe_deserialization()
                        except Exception:
                            pass
                    model = _maybe_in_strategy(lambda: tf.keras.models.load_model(resolved_model_path, **unsafe_kwargs))
            else:
                raise
        config_candidate = getattr(model, "_pipeline_config", None)
        if config_candidate is not None:
            model_config = validate_config(config_candidate)
            model_config.output_dir = base_config.output_dir
        _maybe_in_strategy(lambda: _ensure_model_compiled(model, model_config))
        return model, model_config, resolved_model_path

    raise RuntimeError(
        "Formato de modelo no soportado. Use un archivo `.keras`/`.h5` o proporcione `weights_path`."
    )


def _prepare_eval_predictions(
    run_dir: Path,
    config: PipelineConfig,
    *,
    device: str,
    batch_size: int | None,
    force_recompute: bool,
    output_dir: Path,
    model_path: Path | None,
    weights_path: Path | None,
    best_metric: str,
) -> tuple[pd.DataFrame, dict[str, float], Path, Optional[str]]:
    predictions_path = run_dir / "final_eval_predictions.csv"
    metrics: dict[str, float] = {}
    default_model_path = run_dir / "model_final.keras"
    requested_model_path = model_path.expanduser() if model_path is not None else None
    requested_weights_path = weights_path.expanduser() if weights_path is not None else None

    if requested_weights_path is not None:
        suffix_lower = requested_weights_path.suffix.lower()
        is_weights = requested_weights_path.name.lower().endswith(".weights.h5")
        if requested_weights_path.is_dir() or (suffix_lower in {".keras", ".h5"} and not is_weights):
            if requested_model_path is not None:
                raise SystemExit("Proporciona solo 'model_path' o 'weights_path', no ambos.")
            requested_model_path = requested_weights_path
            requested_weights_path = None

    auto_model_path = requested_model_path
    auto_weights_path = requested_weights_path
    checkpoint_metric_used: Optional[str] = None
    checkpoint_dirs = _candidate_checkpoint_dirs(config, run_dir)

    if auto_model_path is not None and not auto_model_path.exists():
        print(f"⚠️  Modelo especificado no encontrado ({auto_model_path}); se intentará con checkpoints.")
        auto_model_path = None

    if auto_model_path is None and auto_weights_path is None:
        candidates: list[str] = []

        base_metric = (best_metric or "pr_auc").strip().lower().replace(" ", "_").replace("-", "_")

        def _add_candidate(name: str) -> None:
            name = name.strip().lower().replace(" ", "_").replace("-", "_")
            if name and name not in candidates:
                candidates.append(name)

        _add_candidate(base_metric or "pr_auc")
        if base_metric.startswith("val_") or base_metric.startswith("eval_"):
            _add_candidate(base_metric.split("_", 1)[1])
        if base_metric.endswith("_best"):
            _add_candidate(base_metric[: -len("_best")])
        for fallback in ("pr_auc", "roc_auc", "recall"):
            _add_candidate(fallback)

        for metric_key in candidates:
            best_checkpoint = _resolve_best_checkpoint(checkpoint_dirs, metric_key)
            if best_checkpoint is not None:
                auto_weights_path = best_checkpoint.resolve()
                checkpoint_metric_used = metric_key
                break

        if auto_weights_path is None:
            fallback_checkpoint = _discover_latest_checkpoint(checkpoint_dirs)
            if fallback_checkpoint is not None:
                auto_weights_path = fallback_checkpoint.resolve()
                inferred_metric = _infer_metric_from_checkpoint(fallback_checkpoint)
                if inferred_metric:
                    checkpoint_metric_used = inferred_metric.strip().lower()
                elif checkpoint_metric_used:
                    checkpoint_metric_used = checkpoint_metric_used.strip().lower()
                print(
                    f"ℹ️  Usando checkpoint más reciente {fallback_checkpoint.name}"
                    + (
                        f" (métrica inferida: {checkpoint_metric_used})"
                        if checkpoint_metric_used
                        else ""
                    )
                )

    if auto_weights_path is not None:
        suffix_lower = auto_weights_path.suffix.lower()
        is_weights = auto_weights_path.name.lower().endswith(".weights.h5")
        if auto_weights_path.is_dir() or (suffix_lower in {".keras", ".h5"} and not is_weights):
            auto_model_path = auto_weights_path
            auto_weights_path = None

    resolved_model_path = (auto_model_path or default_model_path).expanduser()
    if resolved_model_path.exists():
        resolved_model_path = resolved_model_path.resolve()
    model_source: Path = auto_weights_path or resolved_model_path

    if predictions_path.exists() and not force_recompute:
        if auto_weights_path == requested_weights_path and auto_model_path == requested_model_path:
            df = pd.read_csv(predictions_path)
            if {"y_true", "y_prob", "record"}.issubset(df.columns):
                return df, metrics, model_source, None

    if auto_weights_path is None and not resolved_model_path.exists():
        raise FileNotFoundError(
            f"No se encontró un modelo serializado ni checkpoints (.weights.h5) en {run_dir}.")

    eval_records, _ = collect_records_for_split(config, "eval")
    if not eval_records:
        raise RuntimeError("El split 'eval' no contiene registros; no es posible generar el benchmark.")

    preprocess_settings = resolve_preprocess_settings(config)
    auto_threshold_bytes = None
    if (
        config.dataset_storage == "auto"
        and config.dataset_auto_memmap_threshold_mb is not None
        and config.dataset_auto_memmap_threshold_mb > 0
    ):
        auto_threshold_bytes = int(config.dataset_auto_memmap_threshold_mb * 1024 * 1024)

    eval_bundle = build_windows_dataset(
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
        sampling_strategy="none",
        sampling_seed=config.seed,
        tfrecord_export=None,
        storage_mode="ram",
        memmap_dir=None,
        memmap_prefix=f"benchmark_{config.condition}",
        auto_memmap_threshold_bytes=auto_threshold_bytes,
        feature_worker_processes=config.feature_worker_processes,
        feature_worker_chunk_size=config.feature_worker_chunk_size,
        feature_parallel_min_windows=config.feature_parallel_min_windows,
        force_memmap_after_build=False,
    )

    model, model_config, model_source = _load_model_from_sources(
        base_config=config,
        eval_bundle=eval_bundle,
        default_model_path=default_model_path,
        model_path=auto_model_path,
        weights_path=auto_weights_path,
        output_dir=output_dir,
        device=device,
    )

    scaler_dirs: list[Path] = [run_dir, *checkpoint_dirs]
    if auto_model_path is not None:
        scaler_dirs.append(auto_model_path.parent)
    if auto_weights_path is not None:
        scaler_dirs.append(auto_weights_path.parent)
    scaler_path = _find_feature_scaler(scaler_dirs)
    scaler = None
    if scaler_path is not None:
        try:
            with scaler_path.open("rb") as fp:
                scaler = pickle.load(fp)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"⚠️  No se pudo cargar el scaler en {scaler_path}: {exc}")
            scaler = None

    eval_batch_size = batch_size or config.batch_size

    def _prepare_cpu_model() -> Model:
        nonlocal model, model_config
        try:
            cpu_model = _clone_model_to_device(model, "CPU", model_config)
        except Exception as clone_error:  # pylint: disable=broad-except
            print("⚠️  No se pudo clonar el modelo a CPU; se recargará desde disco.")
            refreshed_model, refreshed_config, _ = _load_model_from_sources(
                base_config=config,
                eval_bundle=eval_bundle,
                default_model_path=default_model_path,
                model_path=auto_model_path,
                weights_path=auto_weights_path,
                output_dir=output_dir,
                device="CPU",
            )
            model = refreshed_model
            model_config = refreshed_config
            return refreshed_model
        model = cpu_model
        return cpu_model

    raw_metrics, predictions, effective_batch, used_cpu = _evaluate_with_memory_fallback(
        model,
        eval_bundle,
        scaler=scaler,
        initial_batch_size=eval_batch_size,
        prefer_device=device,
        get_cpu_model=_prepare_cpu_model,
    )
    if effective_batch != eval_batch_size:
        print(f"ℹ️  Batch de evaluación ajustado a {effective_batch} por limitaciones de memoria.")
    if used_cpu:
        print("ℹ️  La evaluación final se realizó en CPU debido a memoria insuficiente en GPU.")
    predictions.to_csv(predictions_path, index=False)
    return predictions, raw_metrics, model_source, checkpoint_metric_used


def run_benchmark(options: BenchmarkOptions) -> int:
    run_dir = options.run_dir
    if not run_dir.exists():
        raise SystemExit(f"El directorio {run_dir} no existe.")

    output_dir = run_dir / "benchmark_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_run_config(run_dir)
    device = _ensure_device(options.device)

    predictions_df, eval_metrics, model_source, checkpoint_metric_used = _prepare_eval_predictions(
        run_dir,
        config,
        device=device,
        batch_size=options.batch_size,
        force_recompute=options.force_recompute,
        output_dir=output_dir,
        model_path=options.model_path,
        weights_path=options.weights_path,
        best_metric=options.best_metric,
    )

    history_plots: dict[str, Path] = {}
    history_path = run_dir / "training_history.csv"
    if history_path.exists():
        try:
            history_df = pd.read_csv(history_path)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"⚠️  No se pudo leer 'training_history.csv': {exc}")
        else:
            history_plots = _plot_training_curves(history_df, output_dir)

    y_true = predictions_df["y_true"].to_numpy(dtype=np.int8)
    y_prob = predictions_df["y_prob"].to_numpy(dtype=np.float32)

    thresholds = _threshold_sequence(options.threshold_start, options.threshold_stop, options.threshold_step)
    sweep_rows = []
    for thr in thresholds:
        sweep_rows.append(
            _confusion_metrics(
                y_true,
                y_prob,
                float(thr),
                hop_sec=config.hop_sec,
                window_sec=config.window_sec,
                predictions_df=predictions_df,
            )
        )

    sweep_df = pd.DataFrame(sweep_rows).sort_values("threshold").reset_index(drop=True)
    sweep_df.to_csv(output_dir / "threshold_metrics.csv", index=False)

    best_idx = sweep_df["f1"].idxmax()
    best_threshold_row = sweep_df.loc[best_idx]
    subset = sweep_df[sweep_df["f1"] >= best_threshold_row["f1"] - 1e-6]
    best_idx = subset["false_alarms_per_hour"].idxmin()
    best_threshold_row = sweep_df.loc[best_idx]

    best_threshold = float(best_threshold_row["threshold"])
    tn = int(best_threshold_row["tn"])
    fp = int(best_threshold_row["fp"])
    fn = int(best_threshold_row["fn"])
    tp = int(best_threshold_row["tp"])
    best_fpr = float(best_threshold_row["fpr"])
    best_recall = float(best_threshold_row["recall"])
    best_precision = float(best_threshold_row["precision"])

    confusion_img_path = output_dir / f"confusion_matrix_threshold_{best_threshold:.2f}.png"
    _plot_confusion_matrix(tn, fp, fn, tp, best_threshold, confusion_img_path)

    fpr_curve, tpr_curve, roc_thresholds = roc_curve(y_true, y_prob)
    roc_auc_value = auc(fpr_curve, tpr_curve) if fpr_curve.size else float("nan")
    if not fpr_curve.size:
        roc_threshold_series = np.array([], dtype=np.float32)
    elif roc_thresholds.size == fpr_curve.size:
        roc_threshold_series = roc_thresholds.astype(np.float32, copy=False)
    elif roc_thresholds.size == fpr_curve.size - 1:
        roc_threshold_series = np.concatenate([roc_thresholds, [np.nan]]).astype(np.float32, copy=False)
    else:
        padded = np.full(fpr_curve.shape[0], np.nan, dtype=np.float32)
        limit = min(roc_thresholds.size, padded.size)
        padded[:limit] = roc_thresholds[:limit]
        roc_threshold_series = padded
    roc_df = pd.DataFrame(
        {
            "fpr": fpr_curve,
            "tpr": tpr_curve,
            "threshold": roc_threshold_series,
        }
    )
    roc_df.to_csv(output_dir / "roc_curve.csv", index=False)
    highlight_roc_point: Optional[Tuple[float, float]]
    if np.isfinite(best_fpr) and np.isfinite(best_recall):
        highlight_roc_point = (best_fpr, best_recall)
    else:
        highlight_roc_point = None
    _plot_curve(
        fpr_curve,
        tpr_curve,
        xlabel="FPR",
        ylabel="TPR",
        title=f"ROC (AUC={roc_auc_value:.4f})",
        output_path=output_dir / "roc_curve.png",
        highlight_point=highlight_roc_point,
        highlight_label="Mejor threshold",
        highlight_threshold=best_threshold if highlight_roc_point is not None else None,
    )

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc_value = auc(recall_curve, precision_curve) if recall_curve.size else float("nan")
    if pr_thresholds.size:
        if pr_thresholds.size == precision_curve.size:
            pr_threshold_series = pr_thresholds.astype(np.float32, copy=False)
        else:
            pr_threshold_series = np.concatenate([pr_thresholds, [np.nan]]).astype(np.float32, copy=False)
    else:
        pr_threshold_series = np.full(precision_curve.shape[0], np.nan, dtype=np.float32)
    pr_df = pd.DataFrame(
        {
            "recall": recall_curve,
            "precision": precision_curve,
            "threshold": pr_threshold_series,
        }
    )
    pr_df.to_csv(output_dir / "pr_curve.csv", index=False)
    highlight_pr_point: Optional[Tuple[float, float]]
    if np.isfinite(best_recall) and np.isfinite(best_precision):
        highlight_pr_point = (best_recall, best_precision)
    else:
        highlight_pr_point = None
    _plot_curve(
        recall_curve,
        precision_curve,
        xlabel="Recall",
        ylabel="Precision",
        title=f"Precision-Recall (AUC={pr_auc_value:.4f})",
        output_path=output_dir / "pr_curve.png",
        highlight_point=highlight_pr_point,
        highlight_label="Mejor threshold",
        highlight_threshold=best_threshold if highlight_pr_point is not None else None,
    )

    _plot_threshold_sweep(
        sweep_df["threshold"].to_numpy(),
        sweep_df["f1"].to_numpy(),
        sweep_df["false_alarms_per_hour"].to_numpy(),
        output_dir / "threshold_sweep.png",
    )

    summary = {
        "run_dir": str(run_dir),
        "device": device,
        "requested_best_metric": options.best_metric,
        "thresholds": {
            "start": float(thresholds[0]) if thresholds.size else float("nan"),
            "stop": float(thresholds[-1]) if thresholds.size else float("nan"),
            "step": float(options.threshold_step),
            "best_threshold": best_threshold,
        },
        "best_threshold_metrics": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "precision": best_threshold_row["precision"],
            "recall": best_threshold_row["recall"],
            "specificity": best_threshold_row["specificity"],
            "f1": best_threshold_row["f1"],
            "false_alarm_events": best_threshold_row["false_alarm_events"],
            "false_alarms_per_hour": best_threshold_row["false_alarms_per_hour"],
        },
        "roc_auc": float(roc_auc_value),
        "pr_auc": float(pr_auc_value),
        "model_source": str(model_source),
        "confusion_matrix_path": str(confusion_img_path),
    }
    if eval_metrics:
        summary["baseline_eval_metrics"] = {k: float(v) for k, v in eval_metrics.items()}
    if checkpoint_metric_used:
        summary["model_checkpoint_metric"] = checkpoint_metric_used
    if history_plots:
        summary["training_history_plots"] = {key: str(path) for key, path in history_plots.items()}

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("\n✅ Benchmark completado (TensorFlow)")
    print(f"   ↳ Resultados guardados en {output_dir}")
    print(f"   ↳ Mejor threshold sugerido: {best_threshold:.2f} (F1={best_threshold_row['f1']:.4f}, FA/h={best_threshold_row['false_alarms_per_hour']:.4f})")
    if checkpoint_metric_used:
        print(f"   ↳ Checkpoint usado ({checkpoint_metric_used}): {model_source}")
    else:
        print(f"   ↳ Modelo usado: {model_source}")
    if history_plots:
        print("   ↳ Curvas de entrenamiento: " + ", ".join(path.name for path in history_plots.values()))

    return 0


def main(argv: list[str] | None = None) -> int:
    cli_args = sys.argv[1:] if argv is None else argv
    options = _parse_cli_options(cli_args)
    return run_benchmark(options)


if __name__ == "__main__":
    raise SystemExit(main())