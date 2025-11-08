import importlib
import io
import json
import math
import os
import pickle
import sys
import tempfile
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyedflib
from fastapi import HTTPException, UploadFile

from classes.models import (
    EvaluationSummary,
    NetworkEvaluationResponse,
    PredictionInterval,
    SeizureEvent,
)

# -----------------------------------------------------------------------------
# Constants & helpers
# -----------------------------------------------------------------------------

SUPPORTED_BACKENDS = {"tf", "pt"}

MONTAGE_PAIRS = {
    "ar": [
        ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
        ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
        ("A1", "T3"), ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"),
        ("C4", "T4"), ("T4", "A2"), ("FP1", "F3"), ("F3", "C3"),
        ("C3", "P3"), ("P3", "O1"), ("FP2", "F4"), ("F4", "C4"),
        ("C4", "P4"), ("P4", "O2"),
    ],
    "le": [
        ("F7", "F8"), ("T3", "T4"), ("T5", "T6"),
        ("C3", "C4"), ("P3", "P4"), ("O1", "O2"),
    ],
}

EPS = 1e-8


@dataclass(frozen=True)
class BackendPaths:
    base_dir: Path
    config_path: Path
    scaler_path: Path
    model_path: Path


BACKEND_PATHS = {
    "tf": BackendPaths(
        base_dir=Path("networks/tf"),
        config_path=Path("networks/tf/config_used.json"),
        scaler_path=Path("networks/tf/feature_scaler.pkl"),
        model_path=Path("networks/tf/model_final.keras"),
    ),
    "pt": BackendPaths(
        base_dir=Path("networks/pt"),
        config_path=Path("networks/pt/config_used.json"),
        scaler_path=Path("networks/pt/feature_scaler.pkl"),
        model_path=Path("networks/pt/model_final.pt"),
    ),
}

def _resolve_backend_paths(backend: str) -> BackendPaths:
    return BACKEND_PATHS[backend]


def _infer_model_architecture(config: dict, fallback: str) -> str:
    name = config.get("model") or config.get("model_type") or fallback
    if isinstance(name, str) and name.strip():
        return name.strip().lower()
    return fallback


# -----------------------------------------------------------------------------
# EDF loading & preprocessing
# -----------------------------------------------------------------------------


def _detect_suffix(channel_names: Sequence[str]) -> str:
    suffix_candidates = {"-REF", "-LE"}
    counts = {suffix: 0 for suffix in suffix_candidates}
    for name in channel_names:
        for suffix in suffix_candidates:
            if name.endswith(suffix):
                counts[suffix] += 1
    if counts["-REF"] >= counts["-LE"]:
        return "-REF"
    return "-LE"


def _resample_signal(signal: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    if original_fs <= 0 or target_fs <= 0:
        raise ValueError("Las frecuencias de muestreo deben ser positivas.")
    if abs(original_fs - target_fs) < 1e-6:
        return signal.astype(np.float32)
    duration = len(signal) / original_fs
    if duration <= 0:
        return np.zeros(0, dtype=np.float32)
    target_len = max(1, int(round(duration * target_fs)))
    t_original = np.linspace(0.0, duration, num=len(signal), endpoint=False, dtype=np.float64)
    t_target = np.linspace(0.0, duration, num=target_len, endpoint=False, dtype=np.float64)
    resampled = np.interp(t_target, t_original, signal.astype(np.float64))
    return resampled.astype(np.float32)


def _load_bipolar_signals(
    edf_path: Path,
    *,
    montage: str,
    target_fs: float,
) -> Tuple[np.ndarray, float]:
    if montage not in MONTAGE_PAIRS:
        raise HTTPException(status_code=400, detail=f"Montaje '{montage}' no soportado.")

    try:
        reader = pyedflib.EdfReader(str(edf_path))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"No se pudo leer el EDF: {exc}") from exc

    try:
        channel_map: dict[str, Tuple[np.ndarray, float]] = {}
        for idx in range(reader.signals_in_file):
            label = reader.getLabel(idx).strip()
            if not label:
                continue
            norm = label.upper()
            if norm.startswith("EEG "):
                norm = norm[4:]
            freq = float(reader.getSampleFrequency(idx) or 0.0)
            if freq <= 5.0:  # descartamos canales de baja frecuencia (IBI, BURSTS, etc.)
                continue
            signal = reader.readSignal(idx).astype(np.float32)
            channel_map[norm] = (signal, freq)
    finally:
        reader.close()

    if not channel_map:
        raise HTTPException(status_code=400, detail="No se encontraron canales EEG válidos en el EDF.")

    suffix = _detect_suffix(channel_map.keys())
    pairs = MONTAGE_PAIRS[montage]

    bipolar_signals: List[np.ndarray] = []
    freqs: List[float] = []
    for anode, cathode in pairs:
        key_a = f"{anode}{suffix}"
        key_b = f"{cathode}{suffix}"
        if key_a not in channel_map or key_b not in channel_map:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"El EDF no contiene los electrodos requeridos ({anode}/{cathode}) "
                    f"para el montaje '{montage}'."
                ),
            )
        signal_a, freq_a = channel_map[key_a]
        signal_b, freq_b = channel_map[key_b]
        freq = float(freq_a)
        if abs(freq_a - freq_b) > 1e-3:
            raise HTTPException(
                status_code=400,
                detail="Los electrodos emparejados presentan frecuencias de muestreo distintas.",
            )
        bipolar = signal_a.astype(np.float32) - signal_b.astype(np.float32)
        bipolar_signals.append(bipolar)
        freqs.append(freq)

    if not bipolar_signals:
        raise HTTPException(status_code=400, detail="No se pudieron construir canales bipolares del EDF.")

    base_fs = float(np.median(freqs))
    resampled_channels: List[np.ndarray] = []
    min_length: Optional[int] = None
    for channel, freq in zip(bipolar_signals, freqs):
        aligned = _resample_signal(channel, freq, target_fs)
        if min_length is None or aligned.shape[0] < min_length:
            min_length = aligned.shape[0]
        resampled_channels.append(aligned)

    if min_length is None or min_length == 0:
        raise HTTPException(status_code=400, detail="El EDF no contiene muestras suficientes tras el preprocesado.")

    trimmed = [channel[:min_length] for channel in resampled_channels]
    matrix = np.stack(trimmed, axis=1)  # (samples, channels)

    # Normalización z-score por canal
    mean = np.mean(matrix, axis=0, keepdims=True)
    std = np.std(matrix, axis=0, keepdims=True)
    matrix = (matrix - mean) / (std + EPS)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return matrix, target_fs


# -----------------------------------------------------------------------------
# Feature computation (adapted from dataset pipeline)
# -----------------------------------------------------------------------------


def _compute_spectral_features(eeg_tc: np.ndarray, fs: float) -> dict[str, float]:
    T, C = eeg_tc.shape
    if T < 4:
        zeros = {
            "bp_delta": 0.0,
            "bp_theta": 0.0,
            "bp_alpha": 0.0,
            "bp_beta": 0.0,
            "bp_gamma": 0.0,
            "bp_rel_delta": 0.0,
            "bp_rel_theta": 0.0,
            "bp_rel_alpha": 0.0,
            "bp_rel_beta": 0.0,
            "bp_rel_gamma": 0.0,
            "spectral_entropy": 0.0,
            "sef95": 0.0,
        }
        return zeros

    win = np.hanning(T).astype(np.float32)
    xw = eeg_tc * win[:, None]
    Xf = np.fft.rfft(xw, axis=0)
    pxx = np.abs(Xf) ** 2
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)

    def band_power(f_lo: float, f_hi: float) -> float:
        idx = np.where((freqs >= f_lo) & (freqs < f_hi))[0]
        if idx.size == 0:
            return 0.0
        band = pxx[idx, :]
        return float(np.mean(np.sum(band, axis=0)))

    bp_delta = band_power(0.5, 4.0)
    bp_theta = band_power(4.0, 8.0)
    bp_alpha = band_power(8.0, 13.0)
    bp_beta = band_power(13.0, 30.0)
    bp_gamma = band_power(30.0, min(fs / 2.0, 80.0))

    total_idx = np.where((freqs >= 0.5) & (freqs < min(fs / 2.0, 45.0)))[0]
    if total_idx.size == 0:
        total_idx = np.arange(pxx.shape[0])

    total_band = pxx[total_idx, :]
    total_power = np.sum(total_band, axis=0)
    p_norm = total_band / (total_power[None, :] + EPS)
    ent = -np.sum(p_norm * np.log(p_norm + EPS), axis=0)
    K = total_band.shape[0]
    spectral_entropy = float(np.mean(ent / (np.log(K + EPS))))

    csum = np.cumsum(total_band, axis=0)
    thresholds = 0.95 * (total_power + EPS)
    mask95 = csum >= thresholds[None, :]
    idx95 = np.array([
        np.argmax(mask95[:, c]) if mask95[:, c].any() else (mask95.shape[0] - 1)
        for c in range(mask95.shape[1])
    ])
    freqs_band = freqs[total_idx]
    sef95 = float(np.mean(freqs_band[idx95]))

    total_scalar = float(np.mean(total_power))
    return {
        "bp_delta": bp_delta,
        "bp_theta": bp_theta,
        "bp_alpha": bp_alpha,
        "bp_beta": bp_beta,
        "bp_gamma": bp_gamma,
        "bp_rel_delta": bp_delta / (total_scalar + EPS),
        "bp_rel_theta": bp_theta / (total_scalar + EPS),
        "bp_rel_alpha": bp_alpha / (total_scalar + EPS),
        "bp_rel_beta": bp_beta / (total_scalar + EPS),
        "bp_rel_gamma": bp_gamma / (total_scalar + EPS),
        "spectral_entropy": spectral_entropy,
        "sef95": sef95,
    }


def _compute_feature_vector(eeg_tc: np.ndarray, fs: float) -> dict[str, float]:
    x = eeg_tc.astype(np.float32)
    T, C = x.shape
    feats: dict[str, float] = {}

    feats["rms_eeg"] = float(np.sqrt(np.mean(x ** 2)))
    mean_all = float(np.mean(x))
    feats["mad_eeg"] = float(np.mean(np.abs(x - mean_all)))
    feats["std_eeg"] = float(np.mean(np.std(x, axis=0, ddof=0)))

    if T >= 2:
        diff = np.abs(np.diff(x, axis=0))
        feats["line_length"] = float(np.mean(np.sum(diff, axis=0)))
        sgn = np.sign(x)
        sgn[sgn == 0] = 1
        zc = sgn[1:] * sgn[:-1] < 0
        feats["zcr"] = float(np.mean(np.mean(zc.astype(np.float32), axis=0)))
    else:
        feats["line_length"] = 0.0
        feats["zcr"] = 0.0

    if T >= 3:
        tkeo = x[1:-1] ** 2 - x[:-2] * x[2:]
        feats["tkeo_mean"] = float(np.mean(np.mean(tkeo, axis=0)))
    else:
        feats["tkeo_mean"] = 0.0

    if T >= 3:
        xc = x - np.mean(x, axis=0, keepdims=True)
        var0 = np.mean(xc ** 2, axis=0)
        dx = np.diff(xc, axis=0)
        var1 = np.mean(dx ** 2, axis=0)
        mobility = np.sqrt((var1 + EPS) / (var0 + EPS))
        ddx = np.diff(dx, axis=0)
        var2 = np.mean(ddx ** 2, axis=0)
        complexity = np.sqrt((var2 + EPS) / (var1 + EPS)) / (mobility + EPS)
        feats["hjorth_activity"] = float(np.mean(var0))
        feats["hjorth_mobility"] = float(np.mean(mobility))
        feats["hjorth_complexity"] = float(np.mean(complexity))
    else:
        feats["hjorth_activity"] = 0.0
        feats["hjorth_mobility"] = 0.0
        feats["hjorth_complexity"] = 0.0

    spec = _compute_spectral_features(x, fs=fs)
    feats.update(spec)

    bp_alpha = spec["bp_alpha"] + EPS
    bp_beta = spec["bp_beta"] + EPS
    bp_theta = spec["bp_theta"] + EPS

    feats["beta_alpha_ratio"] = spec["bp_beta"] / bp_alpha
    feats["theta_alpha_ratio"] = spec["bp_theta"] / bp_alpha
    feats["ratio_theta_alpha_over_beta"] = (spec["bp_theta"] + spec["bp_alpha"]) / bp_beta
    feats["ratio_theta_beta"] = spec["bp_theta"] / bp_beta
    feats["ratio_theta_alpha_over_alpha_beta"] = (
        (spec["bp_theta"] + spec["bp_alpha"]) / (spec["bp_alpha"] + spec["bp_beta"] + EPS)
    )
    return feats


def _iter_windows(
    signals: np.ndarray,
    *,
    fs: float,
    window_sec: float,
    hop_sec: float,
    window_samples: int,
    hop_samples: int,
):
    if window_samples <= 0 or hop_samples <= 0:
        raise HTTPException(status_code=400, detail="Los parámetros de ventana son inválidos.")
    total_samples = signals.shape[0]
    if window_samples > total_samples:
        raise HTTPException(status_code=400, detail="El EDF es demasiado corto para generar ventanas.")

    produced = False
    idx = 0
    while idx + window_samples <= total_samples:
        segment = np.ascontiguousarray(signals[idx : idx + window_samples], dtype=np.float32)
        start_time = float(idx) / fs
        end_time = start_time + float(window_sec)
        produced = True
        yield segment, start_time, end_time
        idx += hop_samples

    if not produced:
        raise HTTPException(status_code=400, detail="No se pudieron generar ventanas del EDF.")


# -----------------------------------------------------------------------------
# Windowing utilities
# -----------------------------------------------------------------------------


def _build_windows(
    signals: np.ndarray,
    *,
    fs: float,
    window_sec: float,
    hop_sec: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_samples = signals.shape[0]
    window_samples = int(round(window_sec * fs))
    hop_samples = max(1, int(round(hop_sec * fs)))
    if window_samples <= 0 or window_samples > total_samples:
        raise HTTPException(status_code=400, detail="El EDF es demasiado corto para generar ventanas.")

    starts: List[int] = []
    windows: List[np.ndarray] = []
    idx = 0
    while idx + window_samples <= total_samples:
        segment = signals[idx : idx + window_samples]
        windows.append(segment)
        starts.append(idx)
        idx += hop_samples

    if not windows:
        raise HTTPException(status_code=400, detail="No se pudieron generar ventanas del EDF.")

    window_array = np.stack(windows, axis=0)  # (num_windows, samples, channels)
    start_times = np.asarray(starts, dtype=np.float32) / fs
    end_times = start_times + float(window_sec)
    return window_array.astype(np.float32), start_times, end_times


# -----------------------------------------------------------------------------
# CSV label parsing & metrics
# -----------------------------------------------------------------------------


def _parse_events_from_csv(csv_bytes: bytes) -> List[Tuple[float, float]]:
    if not csv_bytes:
        return []
    try:
        df = pd.read_csv(io.StringIO(csv_bytes.decode("utf-8-sig")), comment="#")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el CSV adjunto: {exc}") from exc
    if df.empty:
        return []
    intervals: List[Tuple[float, float]] = []
    for _, row in df.iterrows():
        label = str(row.get("label", "")).lower()
        if "seiz" not in label:
            continue
        start = float(row.get("start_time", 0.0))
        stop = float(row.get("stop_time", 0.0))
        if math.isfinite(start) and math.isfinite(stop) and stop > start:
            intervals.append((start, stop))
    return intervals


def _assign_window_labels(
    start_times: np.ndarray,
    end_times: np.ndarray,
    intervals: Iterable[Tuple[float, float]],
) -> np.ndarray:
    labels = np.zeros(start_times.shape[0], dtype=np.int8)
    for idx, (ws, we) in enumerate(zip(start_times, end_times)):
        for is_, ie in intervals:
            overlap = max(0.0, min(we, ie) - max(ws, is_))
            if overlap > 0.0:
                labels[idx] = 1
                break
    return labels


def _evaluate_thresholds(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, EvaluationSummary, np.ndarray]:
    if labels.size == 0:
        raise ValueError("No hay etiquetas para la evaluación.")

    thresholds = np.linspace(0.05, 0.95, 181)
    best_score = (-1.0, -1.0, float("inf"), 0.0)  # (f1, precision, fp, -threshold)
    best_threshold = 0.5
    best_preds = (probs >= best_threshold).astype(int)
    best_metrics = EvaluationSummary(
        threshold=best_threshold,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        accuracy=0.0,
        confusion_matrix=[[0, 0], [0, 0]],
    )

    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        f1 = (2 * precision * recall) / (precision + recall + EPS)
        accuracy = (tp + tn) / (labels.size + EPS)
        score_key = (f1, precision, fp, -thr)
        if score_key > best_score:
            best_score = score_key
            best_threshold = float(thr)
            best_preds = preds
            best_metrics = EvaluationSummary(
                threshold=best_threshold,
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                accuracy=float(accuracy),
                confusion_matrix=[[tn, fp], [fn, tp]],
            )
    return best_threshold, best_metrics, best_preds


def _choose_threshold_without_labels(probs: np.ndarray) -> Tuple[float, np.ndarray]:
    if probs.size == 0:
        return 0.5, np.zeros(0, dtype=int)
    high_quantile = float(np.quantile(probs, 0.95))
    threshold = max(0.5, high_quantile)
    preds = (probs >= threshold).astype(int)
    return threshold, preds


def _predictions_to_intervals(
    preds: np.ndarray,
    probs: np.ndarray,
    start_times: np.ndarray,
    end_times: np.ndarray,
) -> List[PredictionInterval]:
    intervals: List[PredictionInterval] = []
    idx = 0
    while idx < preds.size:
        if preds[idx] == 0:
            idx += 1
            continue
        start_time = float(start_times[idx])
        scores: List[float] = []
        count = 0
        while idx < preds.size and preds[idx] == 1:
            scores.append(float(probs[idx]))
            count += 1
            end_time = float(end_times[idx])
            idx += 1
        mean_score = float(np.mean(scores)) if scores else 0.0
        intervals.append(
            PredictionInterval(
                start_time=start_time,
                stop_time=end_time,
                score=mean_score,
                count=count,
            )
        )
    return intervals


def _merge_prediction_intervals(
    intervals: List[PredictionInterval],
    *,
    adjacency_tol: float = 1e-6,
) -> List[PredictionInterval]:
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda it: it.start_time)
    merged: List[PredictionInterval] = []

    current_start = sorted_intervals[0].start_time
    current_stop = sorted_intervals[0].stop_time
    current_count = max(0, sorted_intervals[0].count)
    weighted_sum = sorted_intervals[0].score * max(1, sorted_intervals[0].count)
    weight_total = max(1, sorted_intervals[0].count)

    for interval in sorted_intervals[1:]:
        weight = max(1, interval.count)
        if interval.start_time <= current_stop + adjacency_tol:
            current_stop = max(current_stop, interval.stop_time)
            weighted_sum += interval.score * weight
            weight_total += weight
            current_count += max(0, interval.count)
        else:
            merged.append(
                PredictionInterval(
                    start_time=current_start,
                    stop_time=current_stop,
                    score=weighted_sum / weight_total,
                    count=current_count,
                )
            )
            current_start = interval.start_time
            current_stop = interval.stop_time
            current_count = max(0, interval.count)
            weighted_sum = interval.score * weight
            weight_total = weight

    merged.append(
        PredictionInterval(
            start_time=current_start,
            stop_time=current_stop,
            score=weighted_sum / weight_total,
            count=current_count,
        )
    )

    return merged


# -----------------------------------------------------------------------------
# Artifact loaders
# -----------------------------------------------------------------------------


@dataclass
class BackendArtifacts:
    config: dict
    scaler: Optional[object]
    model_inputs: int


@lru_cache(maxsize=None)
def _load_config_cached(backend: str) -> dict:
    paths = _resolve_backend_paths(backend)
    if not paths.config_path.exists():
        raise HTTPException(status_code=500, detail=f"No se encontró la configuración para el backend '{backend}'.")
    try:
        config = json.loads(paths.config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al leer la configuración de '{backend}': {exc}") from exc

    if backend == "tf":
        return _normalize_tf_config(config)
    return config


def _load_config(backend: str) -> dict:
    return _load_config_cached(backend)


def _normalize_tf_config(config: dict) -> dict:
    normalized = dict(config)

    if "window_sec" not in normalized:
        win_val = normalized.get("WINDOW_SEC")
        if win_val is not None:
            try:
                normalized["window_sec"] = float(win_val)
            except (TypeError, ValueError):
                pass

    if "hop_sec" not in normalized:
        hop_val = normalized.get("FRAME_HOP_SEC")
        if hop_val is not None:
            try:
                normalized["hop_sec"] = float(hop_val)
            except (TypeError, ValueError):
                pass

    preprocess = normalized.get("PREPROCESS") if isinstance(normalized.get("PREPROCESS"), dict) else None
    if "target_fs" not in normalized and preprocess:
        resample = preprocess.get("resample") if "resample" in preprocess else preprocess.get("RESAMPLE")
        if resample is not None:
            try:
                normalized["target_fs"] = float(resample)
            except (TypeError, ValueError):
                pass

    normalized.setdefault("target_fs", 256.0)
    normalized.setdefault("window_sec", 10.0)
    normalized.setdefault("hop_sec", 5.0)

    montage_val = normalized.get("montage", normalized.get("MONTAGE"))
    if isinstance(montage_val, str) and montage_val.strip():
        normalized["montage"] = montage_val.strip().lower()
    else:
        normalized["montage"] = "ar"

    if "model" not in normalized:
        model_raw = normalized.get("MODEL")
        if isinstance(model_raw, str):
            alias = model_raw.strip().lower()
            alias = {
                "hyb": "hybrid",
                "hybrid": "hybrid",
                "transformer": "transformer",
                "trf": "transformer",
                "tcn": "tcn",
            }.get(alias, alias)
            normalized["model"] = alias

    if "model_type" not in normalized and "model" in normalized:
        normalized["model_type"] = normalized["model"]

    features = normalized.get("selected_features")
    if not features:
        features = normalized.get("FEATURE_NAMES")
    if isinstance(features, (list, tuple)):
        normalized["selected_features"] = [str(f) for f in features]
    else:
        normalized["selected_features"] = []

    batch_val = normalized.get("eval_batch_size")
    if batch_val is None:
        candidate = normalized.get("BATCH_SIZE", normalized.get("batch_size"))
        if candidate is not None:
            try:
                normalized["eval_batch_size"] = int(candidate)
            except (TypeError, ValueError):
                pass
    normalized.setdefault("eval_batch_size", 64)

    if "time_step_labels" not in normalized:
        normalized["time_step_labels"] = bool(normalized.get("TIME_STEP", normalized.get("time_step", False)))
    if "one_hot_labels" not in normalized:
        normalized["one_hot_labels"] = bool(normalized.get("ONEHOT", normalized.get("one_hot", False)))

    return normalized


def _ensure_numpy_core_alias() -> bool:
    """Garantiza que numpy._core apunte a numpy.core en entornos con NumPy < 2.0."""

    try:
        import numpy  # type: ignore
    except ImportError:
        return False

    try:
        importlib.import_module("numpy._core")
        return True
    except ModuleNotFoundError:
        pass

    try:
        core_pkg = importlib.import_module("numpy.core")
    except ModuleNotFoundError:
        return False

    sys.modules.setdefault("numpy._core", core_pkg)
    for submodule in ("_multiarray_umath", "multiarray", "umath", "numerictypes"):
        try:
            sys.modules[f"numpy._core.{submodule}"] = importlib.import_module(f"numpy.core.{submodule}")
        except ModuleNotFoundError:
            continue
    return True


def _prepare_tf_custom_objects(architecture: str) -> dict[str, object]:
    mapping = {
        "transformer": {
            "module": "tf.models.Transformer",
            "alias": "Transformer",
            "names": [
                "FiLM1D",
                "MultiHeadSelfAttentionRoPE",
                "AttentionPooling1D",
                "AddCLSToken",
                "se_block_1d",
                "gelu",
                "transformer_block_rope",
            ],
        },
        "tcn": {
            "module": "tf.models.TCN",
            "alias": "TCN",
            "names": [
                "FiLM1D",
                "se_block_1d",
                "gated_res_block",
                "causal_sepconv1d",
                "gelu",
            ],
        },
        "hybrid": {
            "module": "tf.models.Hybrid",
            "alias": "Hybrid",
            "names": [
                "FiLM1D",
                "AttentionPooling1D",
                "se_block_1d",
                "gelu",
            ],
        },
    }

    entry = mapping.get(architecture, mapping["transformer"])
    module = importlib.import_module(entry["module"])

    models_pkg = sys.modules.get("models")
    if models_pkg is None:
        models_pkg = types.ModuleType("models")
        sys.modules["models"] = models_pkg
    sys.modules[f"models.{entry['alias']}"] = module
    setattr(models_pkg, entry["alias"], module)

    custom_objects: dict[str, object] = {}
    for name in entry["names"]:
        attr = getattr(module, name, None)
        if attr is not None:
            custom_objects[name] = attr
    return custom_objects


def _coerce_optional_int(value: Optional[object]) -> Optional[int]:
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_tuple_ints(value: Optional[object], fallback: Tuple[int, ...]) -> Tuple[int, ...]:
    if value is None:
        return fallback
    if isinstance(value, (list, tuple)):
        try:
            return tuple(int(v) for v in value)
        except (TypeError, ValueError):
            return fallback
    return fallback


@lru_cache(maxsize=None)
def _load_scaler_cached(backend: str):
    paths = _resolve_backend_paths(backend)
    if not paths.scaler_path.exists():
        return None
    try:
        with paths.scaler_path.open("rb") as fp:
            return pickle.load(fp)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None)
        retry_exc = exc
        missing_str = str(missing) if isinstance(missing, str) else "scikit-learn"
        if isinstance(missing, str) and missing.startswith("numpy._core"):
            if _ensure_numpy_core_alias():
                try:
                    with paths.scaler_path.open("rb") as fp:
                        return pickle.load(fp)
                except ModuleNotFoundError as retry_exc:
                    missing = getattr(retry_exc, "name", missing)
                    missing_str = str(missing) if isinstance(missing, str) else missing_str
        advice = (
            "Verifica que el servidor se ejecute con el mismo entorno virtual donde se instaló scikit-learn "
            "(por ejemplo, usando ./.venv/bin/uvicorn) o instala el paquete con 'python3 -m pip install scikit-learn'."
        )
        if isinstance(missing, str) and missing.startswith("numpy._core"):
            advice += (
                " También puedes actualizar NumPy a la versión 2.0 o superior en el mismo entorno"
                " ('python3 -m pip install \"numpy>=2.0\"') para alinear el formato del pickle."
            )
        raise HTTPException(
            status_code=500,
            detail=(
                "No se pudo cargar el scaler porque falta un módulo requerido: "
                f"'{missing_str}'. {advice}"
            ),
        ) from retry_exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Error al cargar el scaler del backend '{backend}': {exc}") from exc


def _load_scaler(backend: str):
    return _load_scaler_cached(backend)


@lru_cache(maxsize=None)
def _load_tf_model_cached() -> Tuple[object, int]:
    paths = _resolve_backend_paths("tf")
    if not paths.model_path.exists():
        raise HTTPException(status_code=500, detail="No se encontró el modelo TensorFlow preentrenado.")
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(
            status_code=500,
            detail=(
                "TensorFlow no está instalado en el servidor. "
                "Instala tensorflow>=2.10 para habilitar eval_network_tf."
            ),
        ) from exc

    config = _load_config("tf")
    architecture = _infer_model_architecture(config, "transformer")
    custom_objects = _prepare_tf_custom_objects(architecture)

    try:
        model = tf.keras.models.load_model(
            str(paths.model_path),
            compile=False,
            custom_objects=custom_objects or None,
            safe_mode=False,
        )
        input_count = len(model.inputs) if isinstance(model.inputs, list) else 1
        return model, input_count
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Error al cargar el modelo TensorFlow: {exc}") from exc


def _load_tf_model() -> Tuple[object, int]:
    return _load_tf_model_cached()


def _load_pt_model(input_shape: Tuple[int, int], feat_dim: int, config: dict):
    paths = _resolve_backend_paths("pt")
    if not paths.model_path.exists():
        raise HTTPException(status_code=500, detail="No se encontró el modelo PyTorch preentrenado.")
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(
            status_code=500,
            detail=(
                "PyTorch no está instalado en el servidor. "
                "Instala torch>=2.0 para habilitar eval_network_pt."
            ),
        ) from exc

    architecture = _infer_model_architecture(config, "transformer")
    feat_input_dim = feat_dim if feat_dim > 0 else None
    time_step = bool(config.get("time_step_labels", False))
    one_hot = bool(config.get("one_hot_labels", False))

    try:
        if architecture == "transformer":
            from pt.models.Transformer import build_transformer  # type: ignore

            model = build_transformer(
                input_shape=input_shape,
                num_classes=1,
                embed_dim=int(config.get("transformer_embed_dim", 128)),
                num_layers=int(config.get("transformer_num_layers", 4)),
                num_heads=int(config.get("transformer_num_heads", 4)),
                mlp_dim=int(config.get("transformer_mlp_dim", 256)),
                dropout_rate=float(
                    config.get("transformer_dropout_rate", config.get("transformer_dropout", config.get("dropout", 0.1)))
                ),
                time_step_classification=time_step,
                one_hot=one_hot,
                use_se=bool(config.get("transformer_use_se", False)),
                se_ratio=int(config.get("transformer_se_ratio", 16)),
                feat_input_dim=feat_input_dim,
                koopman_latent_dim=int(config.get("transformer_koopman_latent_dim", 0)),
                koopman_loss_weight=float(config.get("transformer_koopman_loss_weight", 0.0)),
                use_reconstruction_head=bool(config.get("transformer_use_reconstruction_head", False)),
                recon_weight=float(config.get("transformer_recon_weight", 0.0)),
                recon_target=str(config.get("transformer_recon_target", "signal")),
                bottleneck_dim=_coerce_optional_int(config.get("transformer_bottleneck_dim")),
                expand_dim=_coerce_optional_int(config.get("transformer_expand_dim")),
            )
        elif architecture == "tcn":
            from pt.models.TCN import build_tcn  # type: ignore

            model = build_tcn(
                input_shape,
                num_classes=1,
                num_filters=int(config.get("tcn_num_filters", config.get("num_filters", 64))),
                kernel_size=int(config.get("tcn_kernel_size", config.get("kernel_size", 7))),
                dropout_rate=float(config.get("tcn_dropout_rate", config.get("dropout", 0.25))),
                num_blocks=int(config.get("tcn_num_blocks", 8)),
                time_step_classification=time_step,
                one_hot=one_hot,
                separable=bool(config.get("tcn_separable", False)),
                se_ratio=int(config.get("tcn_se_ratio", config.get("transformer_se_ratio", 16))),
                cycle_dilations=_coerce_tuple_ints(config.get("tcn_cycle_dilations"), (1, 2, 4, 8)),
                feat_input_dim=feat_input_dim,
                use_attention_pool_win=bool(config.get("tcn_use_attention_pool_win", True)),
                koopman_latent_dim=int(config.get("tcn_koopman_latent_dim", config.get("transformer_koopman_latent_dim", 0))),
                koopman_loss_weight=float(config.get("tcn_koopman_loss_weight", config.get("transformer_koopman_loss_weight", 0.0))),
                use_reconstruction_head=bool(
                    config.get("tcn_use_reconstruction_head", config.get("transformer_use_reconstruction_head", False))
                ),
                recon_weight=float(config.get("tcn_recon_weight", config.get("transformer_recon_weight", 0.0))),
                recon_target=str(config.get("tcn_recon_target", config.get("transformer_recon_target", "signal"))),
                bottleneck_dim=_coerce_optional_int(
                    config.get("tcn_bottleneck_dim", config.get("transformer_bottleneck_dim"))
                ),
                expand_dim=_coerce_optional_int(config.get("tcn_expand_dim", config.get("transformer_expand_dim"))),
            )
        elif architecture == "hybrid":
            from pt.models.Hybrid import build_hybrid  # type: ignore

            model = build_hybrid(
                input_shape=input_shape,
                num_classes=1,
                one_hot=one_hot,
                time_step=time_step,
                conv_type=str(config.get("hybrid_conv_type", "conv")),
                num_filters=int(config.get("hybrid_num_filters", config.get("num_filters", 64))),
                kernel_size=int(config.get("hybrid_kernel_size", config.get("kernel_size", 7))),
                se_ratio=int(config.get("hybrid_se_ratio", 16)),
                dropout_rate=float(config.get("hybrid_dropout_rate", config.get("dropout", 0.25))),
                num_heads=int(config.get("hybrid_num_heads", config.get("transformer_num_heads", 4))),
                rnn_units=int(config.get("hybrid_rnn_units", config.get("rnn_units", 64))),
                feat_input_dim=feat_input_dim,
                use_se_after_cnn=bool(config.get("hybrid_use_se_after_cnn", True)),
                use_se_after_rnn=bool(config.get("hybrid_use_se_after_rnn", True)),
                use_between_attention=bool(config.get("hybrid_use_between_attention", True)),
                use_final_attention=bool(config.get("hybrid_use_final_attention", True)),
                koopman_latent_dim=int(config.get("hybrid_koopman_latent_dim", config.get("transformer_koopman_latent_dim", 0))),
                koopman_loss_weight=float(config.get("hybrid_koopman_loss_weight", config.get("transformer_koopman_loss_weight", 0.0))),
                use_reconstruction_head=bool(
                    config.get("hybrid_use_reconstruction_head", config.get("transformer_use_reconstruction_head", False))
                ),
                recon_weight=float(config.get("hybrid_recon_weight", config.get("transformer_recon_weight", 0.0))),
                recon_target=str(config.get("hybrid_recon_target", config.get("transformer_recon_target", "signal"))),
                bottleneck_dim=_coerce_optional_int(config.get("hybrid_bottleneck_dim", config.get("transformer_bottleneck_dim"))),
                expand_dim=_coerce_optional_int(config.get("hybrid_expand_dim", config.get("transformer_expand_dim"))),
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Arquitectura de modelo PyTorch '{architecture}' no soportada.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo instanciar el modelo PyTorch '{architecture}': {exc}",
        ) from exc

    checkpoint = torch.load(str(paths.model_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict") or checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------


def evaluate_network(edf_path: Path, csv_bytes: Optional[bytes], backend: str) -> NetworkEvaluationResponse:
    backend = backend.lower()
    if backend not in SUPPORTED_BACKENDS:
        raise HTTPException(status_code=400, detail=f"Backend '{backend}' no soportado.")

    config = _load_config(backend)
    model_architecture = _infer_model_architecture(config, backend)
    scaler = _load_scaler(backend)

    window_sec = float(config.get("window_sec", 10.0))
    hop_sec = float(config.get("hop_sec", 5.0))
    target_fs = float(config.get("target_fs", 256.0))
    montage = str(config.get("montage", "ar")).lower()
    selected_features = tuple(config.get("selected_features", []))

    signals, fs = _load_bipolar_signals(edf_path, montage=montage, target_fs=target_fs)
    window_samples = int(round(window_sec * fs))
    hop_samples = max(1, int(round(hop_sec * fs)))
    if window_samples <= 0:
        raise HTTPException(status_code=400, detail="La duración de la ventana configurada es inválida.")

    batch_size = int(config.get("eval_batch_size", 64))
    batch_size = max(1, batch_size)

    feature_names = list(selected_features)
    feature_dim = len(feature_names)
    if feature_dim > 0 and scaler is None:
        raise HTTPException(
            status_code=500,
            detail="El modelo requiere features escaladas pero el scaler no está disponible.",
        )

    def _feature_row(window: np.ndarray) -> List[float]:
        feats = _compute_feature_vector(window, fs=fs)
        return [float(feats.get(name, 0.0)) for name in feature_names]

    start_times_list: List[float] = []
    end_times_list: List[float] = []
    probs_list: List[float] = []

    window_batch: List[np.ndarray] = []
    feature_batch: List[List[float]] = []

    if backend == "tf":
        model, input_count = _load_tf_model()
        import tensorflow as tf  # type: ignore  # noqa: F401

        def _run_batch() -> None:
            if not window_batch:
                return
            win_array = np.stack(window_batch, axis=0).astype(np.float32)
            inputs: object
            if input_count == 2:
                if feature_dim == 0 or not feature_batch:
                    raise HTTPException(
                        status_code=500,
                        detail="El modelo TensorFlow espera features auxiliares pero no se proporcionaron.",
                    )
                feats_np = np.asarray(feature_batch, dtype=np.float32)
                feats_np = scaler.transform(feats_np) if scaler is not None else feats_np
                inputs = [win_array, feats_np]
            else:
                inputs = win_array
            raw_probs = model.predict(inputs, verbose=0)
            prob_arr = np.asarray(raw_probs)
            if prob_arr.ndim > 1:
                prob_arr = prob_arr.reshape(prob_arr.shape[0], -1).mean(axis=1)
            else:
                prob_arr = prob_arr.reshape(-1)
            probs_list.extend(prob_arr.astype(np.float64).tolist())

    else:
        model = _load_pt_model(
            input_shape=(window_samples, signals.shape[1]),
            feat_dim=feature_dim,
            config=config,
        )
        import torch  # type: ignore

        def _run_batch() -> None:
            if not window_batch:
                return
            win_array = np.stack(window_batch, axis=0).astype(np.float32)
            seq_tensor = torch.from_numpy(win_array)
            feat_tensor = None
            if feature_dim > 0:
                if not feature_batch:
                    raise HTTPException(
                        status_code=500,
                        detail="El modelo PyTorch espera features auxiliares pero no se proporcionaron.",
                    )
                feats_np = np.asarray(feature_batch, dtype=np.float32)
                feats_np = scaler.transform(feats_np) if scaler is not None else feats_np
                feat_tensor = torch.from_numpy(feats_np.astype(np.float32))
            with torch.no_grad():
                outputs = model(seq_tensor, feat_tensor) if feat_tensor is not None else model(seq_tensor)
                prob_tensor = outputs.probabilities if hasattr(outputs, "probabilities") else outputs
                prob_arr = prob_tensor.detach().cpu().numpy()
            if prob_arr.ndim > 1:
                prob_arr = prob_arr.reshape(prob_arr.shape[0], -1).mean(axis=1)
            else:
                prob_arr = prob_arr.reshape(-1)
            probs_list.extend(prob_arr.astype(np.float64).tolist())

    for window, start_time, end_time in _iter_windows(
        signals,
        fs=fs,
        window_sec=window_sec,
        hop_sec=hop_sec,
        window_samples=window_samples,
        hop_samples=hop_samples,
    ):
        start_times_list.append(start_time)
        end_times_list.append(end_time)
        window_batch.append(window)
        if feature_dim > 0:
            feature_batch.append(_feature_row(window))
        if len(window_batch) >= batch_size:
            _run_batch()
            window_batch.clear()
            feature_batch.clear()

    if window_batch:
        _run_batch()
        window_batch.clear()
        feature_batch.clear()

    if not probs_list:
        raise HTTPException(status_code=500, detail="No se pudieron generar predicciones para las ventanas del EDF.")

    start_times = np.asarray(start_times_list, dtype=np.float32)
    end_times = np.asarray(end_times_list, dtype=np.float32)
    probs = np.asarray(probs_list, dtype=np.float32)

    if probs.size != start_times.size:
        raise HTTPException(status_code=500, detail="Conteo inconsistente de ventanas al evaluar el modelo.")

    probs = np.clip(probs, 0.0, 1.0)

    csv_intervals = _parse_events_from_csv(csv_bytes) if csv_bytes else []
    metrics: Optional[EvaluationSummary] = None
    predictions: np.ndarray
    threshold: float

    if csv_intervals:
        labels = _assign_window_labels(start_times, end_times, csv_intervals)
        threshold, metrics, predictions = _evaluate_thresholds(probs, labels)
    else:
        threshold, predictions = _choose_threshold_without_labels(probs)

    intervals = _predictions_to_intervals(predictions, probs, start_times, end_times)
    intervals = _merge_prediction_intervals(
        intervals,
        adjacency_tol=max(1e-6, hop_sec * 1e-3),
    )

    events = [
        SeizureEvent(
            channel=None,
            start_time=interval.start_time,
            stop_time=interval.stop_time,
            label="predicted_seizure",
            confidence=interval.score,
        )
        for interval in intervals
    ]

    return NetworkEvaluationResponse(
        model_type=backend,
        model_architecture=model_architecture,
        window_sec=window_sec,
        hop_sec=hop_sec,
        threshold=threshold,
        events=events,
        metrics=metrics,
    )

async def _run_evaluation(
    backend: str,
    edf_file: UploadFile,
    csv_file: Optional[UploadFile],
) -> NetworkEvaluationResponse:
    if not edf_file.filename:
        raise HTTPException(status_code=400, detail="Se requiere un archivo EDF válido.")

    edf_bytes = await edf_file.read()
    if not edf_bytes:
        raise HTTPException(status_code=400, detail="El archivo EDF está vacío.")

    csv_bytes: Optional[bytes] = None
    if csv_file is not None and csv_file.filename:
        content = await csv_file.read()
        if content:
            csv_bytes = content

    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_edf:
        tmp_edf.write(edf_bytes)
        tmp_edf.flush()
        tmp_path = Path(tmp_edf.name)

    try:
        return evaluate_network(tmp_path, csv_bytes, backend=backend)
    finally:
        os.unlink(tmp_path)

