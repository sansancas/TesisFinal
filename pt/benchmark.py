#!/usr/bin/env python
"""Benchmark de métricas sobre el split de evaluación.

Este script carga una corrida entrenada (directorio con ``model_final.pt`` y
``config_used.json``), recupera o genera las predicciones del split ``eval`` y
produce un informe con:

* Matriz de confusión a nivel ventana.
* Curvas ROC y Precision-Recall junto con sus áreas bajo la curva.
* Sweep de umbrales que incluye falsos positivos por hora para cada valor.
* Curvas de entrenamiento (loss y accuracy) extraídas de ``training_history.csv``.
* Gráficos auxiliares y tablas guardadas en ``<run_dir>/benchmark_eval``.

Uso típico::

    python -m pt.benchmark_eval run_dir=runs/mi_experimento/20251005-123456
    python -m pt.benchmark_eval run_dir=runs/mi_experimento/20251005-123456 weights_path=/ruta/pesos.pt

Todos los parámetros se pasan como ``clave=valor`` y el script puede trabajar
tanto con un checkpoint que contenga el modelo completo (``model_path``) como
con archivos de pesos (``weights_path``). Si no se especifica ninguno, intentará
cargar automáticamente el mejor checkpoint disponible para la métrica indicada
con ``best_metric`` (por defecto ``pr_auc``).

"""
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Backend para entornos sin display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from dataset import (
    build_torch_dataset_from_arrays,
    build_windows_dataset,
    collect_records_for_split,
)
from pipeline import (
    _collect_predictions,
    _compute_evaluation_metrics,
    _window_predictions,
    make_model,
)
from utils import PipelineConfig, load_config, resolve_preprocess_settings, validate_config
from torch.utils.data import DataLoader


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
        "Uso: python -m pt.benchmark_eval run_dir=/ruta/al/run "
        "[device=cuda] [threshold_start=0.05] [threshold_stop=0.95] [threshold_step=0.05] "
        "[force_recompute=true] [batch_size=128] [model_path=/otra/ruta/model.pt] "
        "[weights_path=/otra/ruta/weights.pt] [best_metric=pr_auc]"
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
            raise SystemExit(f"Argumento '{token}' inválido. { _usage() }")
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
        best_metric=(best_metric_raw.strip().lower() or "pr_auc"),
    )


def _load_run_config(run_dir: Path) -> PipelineConfig:
    config_path = run_dir / "config_used.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró '{config_path}'.")
    config = load_config(config_path)
    config = validate_config(config)
    # Aseguramos que los artefactos se escriban dentro del run-dir.
    config.output_dir = run_dir
    if config.dataset_memmap_dir is None and config.dataset_storage in {"memmap", "auto"}:
        config.dataset_memmap_dir = run_dir / "memmap"
    return config


def _ensure_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    """Devuelve (n_false_alarms, false_alarms_per_hour)."""
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


def _resolve_best_checkpoint(run_dir: Path, metric: str) -> Path | None:
    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.exists():
        return None

    sanitized_metric = metric.strip().lower()
    candidates = [
        checkpoint_root / "final" / f"final_{sanitized_metric}_best.pt",
        checkpoint_root / f"final_{sanitized_metric}_best.pt",
        checkpoint_root / f"{sanitized_metric}_best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(
        checkpoint_root.rglob(f"*{sanitized_metric}*best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


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
    fig.tight_layout()
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

    if "epoch" not in history_df.columns:
        history_df = history_df.copy()
        history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))

    epochs = history_df["epoch"].to_numpy()

    if "loss" in history_df.columns or "val_loss" in history_df.columns:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        if "loss" in history_df.columns:
            ax.plot(epochs, history_df["loss"], label="Train loss", color="tab:blue")
        if "val_loss" in history_df.columns:
            ax.plot(epochs, history_df["val_loss"], label="Val loss", color="tab:orange")
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

    train_acc_col = next((col for col in history_df.columns if col.lower().startswith("train_") and col.lower().endswith("accuracy")), None)
    val_acc_col = next((col for col in history_df.columns if col.lower().startswith("val_") and col.lower().endswith("accuracy")), None)
    if train_acc_col or val_acc_col:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        if train_acc_col:
            ax.plot(epochs, history_df[train_acc_col], label="Train accuracy", color="tab:green")
        if val_acc_col:
            ax.plot(epochs, history_df[val_acc_col], label="Val accuracy", color="tab:red")
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

    return created


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "weights", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()) and all(
            torch.is_tensor(v) for v in checkpoint.values()
        ):
            return checkpoint
    raise RuntimeError("No se pudo extraer un state_dict del checkpoint proporcionado.")


def _load_model_from_sources(
    base_config: PipelineConfig,
    eval_bundle,
    *,
    device: torch.device,
    default_model_path: Path,
    model_path: Path | None,
    weights_path: Path | None,
    output_dir: Path,
) -> tuple[nn.Module, PipelineConfig, Path]:
    model_config = base_config

    def _config_from_dict(config_like: object) -> PipelineConfig:
        if isinstance(config_like, PipelineConfig):
            cfg = validate_config(config_like)
        elif isinstance(config_like, dict):
            tmp_path = output_dir / "_tmp_model_config.json"
            tmp_path.write_text(json.dumps(config_like, indent=2), encoding="utf-8")
            cfg = validate_config(load_config(tmp_path))
            tmp_path.unlink(missing_ok=True)
        else:
            return base_config
        cfg.output_dir = base_config.output_dir
        if base_config.dataset_memmap_dir is not None and cfg.dataset_memmap_dir is None:
            cfg.dataset_memmap_dir = base_config.dataset_memmap_dir
        return cfg

    if weights_path is not None:
        if not weights_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de pesos: {weights_path}")
        raw_state = torch.load(weights_path, map_location="cpu")
        state_dict = _extract_state_dict(raw_state)
        model = make_model(
            model_type=model_config.model,
            input_shape=eval_bundle.sequences.shape[1:],
            feat_dim=(eval_bundle.features.shape[1] if eval_bundle.features is not None else None),
            num_filters=model_config.num_filters,
            kernel_size=model_config.kernel_size,
            dropout_rate=model_config.dropout,
            rnn_units=model_config.rnn_units,
            time_step=model_config.time_step_labels,
        ).to(device)
        model.load_state_dict(state_dict, strict=True)
        return model, model_config, weights_path

    resolved_model_path = model_path or default_model_path
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {resolved_model_path}")

    checkpoint = torch.load(resolved_model_path, map_location=device)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint.to(device)
        config_candidate = getattr(checkpoint, "_pipeline_config", None)
        if config_candidate is not None:
            model_config = _config_from_dict(config_candidate)
        return model, model_config, resolved_model_path

    config_candidate = None
    if isinstance(checkpoint, dict):
        config_candidate = checkpoint.get("config")
        if config_candidate is not None:
            model_config = _config_from_dict(config_candidate)
        state_dict = _extract_state_dict(checkpoint)
    else:
        state_dict = _extract_state_dict(checkpoint)

    model = make_model(
        model_type=model_config.model,
        input_shape=eval_bundle.sequences.shape[1:],
        feat_dim=(eval_bundle.features.shape[1] if eval_bundle.features is not None else None),
        num_filters=model_config.num_filters,
        kernel_size=model_config.kernel_size,
        dropout_rate=model_config.dropout,
        rnn_units=model_config.rnn_units,
        time_step=model_config.time_step_labels,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    return model, model_config, resolved_model_path


def _run_model_inference(
    model: torch.nn.Module,
    bundle,
    *,
    device: torch.device,
    batch_size: int,
    scaler,
):
    sequences = bundle.sequences
    labels = bundle.labels.astype(np.float32)
    features = bundle.features
    label_mode = getattr(bundle, "label_mode", "window")

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

    prev_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        probs = _collect_predictions(model, dataloader, device)
    finally:
        torch.set_grad_enabled(prev_grad)

    metrics = _compute_evaluation_metrics(probs, labels, label_mode=label_mode)
    metrics["loss"] = 0.0

    predictions = _window_predictions(
        probs,
        labels,
        bundle.patients,
        bundle.records,
        label_mode=label_mode,
    )
    return metrics, predictions


def _prepare_eval_predictions(
    run_dir: Path,
    config: PipelineConfig,
    *,
    device: torch.device,
    batch_size: int | None,
    force_recompute: bool,
    output_dir: Path,
    model_path: Path | None,
    weights_path: Path | None,
    best_metric: str,
) -> tuple[pd.DataFrame, dict[str, float], Path, Optional[str]]:
    predictions_path = run_dir / "final_eval_predictions.csv"
    metrics: dict[str, float] = {}
    default_model_path = run_dir / "model_final.pt"

    auto_model_path = model_path
    auto_weights_path = weights_path
    checkpoint_metric_used: Optional[str] = None

    if auto_weights_path is None and auto_model_path is None:
        # Prefer the best checkpoint for the requested metric when available.
        metric_candidates = []
        sanitized_metric = best_metric.strip().lower() if best_metric else ""
        if sanitized_metric:
            metric_candidates.append(sanitized_metric)
        if "pr_auc" not in metric_candidates:
            metric_candidates.append("pr_auc")
        if "roc_auc" not in metric_candidates:
            metric_candidates.append("roc_auc")
        if "recall" not in metric_candidates:
            metric_candidates.append("recall")
        for metric_key in metric_candidates:
            best_checkpoint = _resolve_best_checkpoint(run_dir, metric_key)
            if best_checkpoint is not None:
                auto_weights_path = best_checkpoint
                checkpoint_metric_used = metric_key
                break

    resolved_model_path = auto_model_path or default_model_path
    model_source: Path = auto_weights_path or resolved_model_path

    # Si ya existe el CSV y no se desea recomputar, lo devolvemos tal cual.
    if (
        predictions_path.exists()
        and not force_recompute
        and auto_weights_path == weights_path
        and auto_model_path == model_path
    ):
        df = pd.read_csv(predictions_path)
        if {"y_true", "y_prob", "record"}.issubset(df.columns):
            return df, metrics, model_source, None
        # Si faltan columnas clave, recomputamos.

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

    model, _model_config, model_source = _load_model_from_sources(
        config,
        eval_bundle,
        device=device,
        default_model_path=default_model_path,
        model_path=auto_model_path,
        weights_path=auto_weights_path,
        output_dir=output_dir,
    )

    scaler_path = run_dir / "feature_scaler.pkl"
    scaler = None
    if scaler_path.exists():
        with scaler_path.open("rb") as fp:
            scaler = pickle.load(fp)

    eval_batch_size = batch_size or config.batch_size
    metrics, predictions = _run_model_inference(
        model,
        eval_bundle,
        device=device,
        batch_size=eval_batch_size,
        scaler=scaler,
    )
    predictions.to_csv(predictions_path, index=False)
    return predictions, metrics, model_source, checkpoint_metric_used


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
    sweep_path = output_dir / "threshold_metrics.csv"
    sweep_df.to_csv(sweep_path, index=False)

    # Seleccionamos el mejor umbral por F1; como tie-breaker, menor falsas alarmas por hora.
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
        highlight_threshold=best_threshold if highlight_roc_point else None,
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
        highlight_threshold=best_threshold if highlight_pr_point else None,
    )

    _plot_threshold_sweep(
        sweep_df["threshold"].to_numpy(),
        sweep_df["f1"].to_numpy(),
        sweep_df["false_alarms_per_hour"].to_numpy(),
        output_dir / "threshold_sweep.png",
    )

    summary = {
        "run_dir": str(run_dir),
        "device": str(device),
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
    }
    if eval_metrics:
        summary["baseline_eval_metrics"] = {k: float(v) for k, v in eval_metrics.items()}
    if checkpoint_metric_used:
        summary["model_checkpoint_metric"] = checkpoint_metric_used
    if history_plots:
        summary["training_history_plots"] = {key: str(path) for key, path in history_plots.items()}

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("\n✅ Benchmark completado")
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
