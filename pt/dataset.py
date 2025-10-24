import os
import gc
import tempfile
from concurrent.futures import ProcessPoolExecutor
from uuid import uuid4
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence
import hashlib
import math

import numpy as np
import mne
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from numpy.lib.stride_tricks import as_strided

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
from utils import (
    apply_preprocess_config,
    PipelineConfig,
    SPLIT_ATTRS,
    PREPROCESS,
    BANDPASS,
    NOTCH,
    NORMALIZE,
    TFRecordExportConfig,
)

# =================================================================================================================
# Montage-based CSV listing & filtering
# =================================================================================================================

def detect_suffix_strict(ch_names: Sequence[str]) -> str:
    """Detect channel name suffix across all channels robustly."""
    le = sum(1 for ch in ch_names if ch.endswith('-LE'))
    rf = sum(1 for ch in ch_names if ch.endswith('-REF'))
    if le > rf and le > 0:
        return '-LE'
    if rf > le and rf > 0:
        return '-REF'
    for suf in ('-LE', '-REF'):
        for base in ('F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2'):
            if any(ch == base + suf for ch in ch_names):
                return suf
    return '-REF'


MONTAGE_PAIRS = {
    'ar': [
        ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
        ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
        ('A1', 'T3'), ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'),
        ('C4', 'T4'), ('T4', 'A2'), ('FP1', 'F3'), ('F3', 'C3'),
        ('C3', 'P3'), ('P3', 'O1'), ('FP2', 'F4'), ('F4', 'C4'),
        ('C4', 'P4'), ('P4', 'O2')
    ],
    'le': [
        ('F7', 'F8'), ('T3', 'T4'), ('T5', 'T6'),
        ('C3', 'C4'), ('P3', 'P4'), ('O1', 'O2')
    ],
}


def _notch_freqs_from_cfg(cfg: dict) -> list[float] | None:
    notch = cfg.get('notch', None)
    if notch in (None, 0, False):
        return None
    if isinstance(notch, (list, tuple)):
        return [float(f) for f in notch if f]
    base = float(notch)
    n_h = int(cfg.get('n_harmonics', 0)) if cfg.get('notch_harmonics', False) else 0
    return [base * (i + 1) for i in range(n_h + 1)]


def preprocess_edf(raw: mne.io.BaseRaw, config: dict = PREPROCESS) -> mne.io.BaseRaw:
    """Preprocess EDF in-place following global flags."""
    if not any([BANDPASS, NOTCH, NORMALIZE]):
        if config.get('resample', 0) and config['resample'] != raw.info['sfreq']:
            raw.resample(config['resample'], npad='auto', verbose=False)
        return raw

    if BANDPASS:
        raw.filter(
            config['bandpass'][0],
            config['bandpass'][1],
            method='fir',
            phase='zero',
            verbose=False,
        )

    if NOTCH:
        freqs = _notch_freqs_from_cfg(config)
        if freqs:
            nyquist = float(raw.info['sfreq']) / 2.0
            max_allowed = nyquist - 1e-6
            freqs = [float(f) for f in freqs if 0.0 < float(f) <= max_allowed]
            if freqs:
                raw.notch_filter(
                    freqs=freqs,
                    method='fir',
                    phase='zero',
                    verbose=False,
                )

    if config.get('resample', 0) and config['resample'] != raw.info['sfreq']:
        raw.resample(config['resample'], npad='auto', verbose=False)

    if NORMALIZE:
        raw.apply_function(
            lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-6),
            picks='eeg',
            channel_wise=False,
        )
    return raw


def extract_montage_signals(edf_path: str, montage: str = 'ar', desired_fs: int = 0) -> mne.io.BaseRaw:
    if montage not in MONTAGE_PAIRS:
        raise ValueError(f"Unsupported montage '{montage}'. Use 'ar' or 'le'.")

    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    ch_names_set = set(raw.ch_names)
    suf = detect_suffix_strict(raw.ch_names)
    pairs = [(f'EEG {a}{suf}', f'EEG {b}{suf}') for a, b in MONTAGE_PAIRS[montage]]
    needed = {c for pair in pairs for c in pair}
    missing = needed - ch_names_set
    if missing:
        raw.close()
        raise RuntimeError(f"Missing required electrodes for {montage} montage: {missing}")

    raw.load_data()
    raw = preprocess_edf(raw, config=PREPROCESS)

    anodes = [a for a, _ in pairs]
    cathodes = [b for _, b in pairs]
    ch_names_bip = [f"{a}-{b}" for a, b in pairs]
    raw_bip = mne.set_bipolar_reference(
        raw,
        anode=anodes,
        cathode=cathodes,
        ch_name=ch_names_bip,
        drop_refs=True,
        verbose=False,
    )
    raw_bip.pick(ch_names_bip)

    if desired_fs > 0 and raw_bip.info['sfreq'] != desired_fs:
        raw_bip.resample(desired_fs, npad='auto', verbose=False)
    return raw_bip


def read_intervals(csv_path: Path) -> list[tuple[float, float]]:
    try:
        df = pd.read_csv(csv_path, comment="#")
    except FileNotFoundError:
        return []
    intervals: list[tuple[float, float]] = []
    for _, row in df.iterrows():
        label = str(row.get("label", "")).lower()
        if "seiz" in label:
            intervals.append((float(row["start_time"]), float(row["stop_time"])))
    return intervals


def consolidate_records(primary: Iterable[Path], fallback: Iterable[Path], *, max_records: int, max_per_patient: int) -> list[Path]:
    selected: list[Path] = []
    seen: set[str] = set()
    patient_counts: dict[str, int] = defaultdict(int)
    for candidate in list(primary) + list(fallback):
        if candidate is None:
            continue
        path = Path(candidate)
        if not path.exists():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        patient_id = path.stem.split("_")[0]
        if max_per_patient and patient_counts[patient_id] >= max_per_patient:
            continue
        selected.append(path)
        seen.add(key)
        patient_counts[patient_id] += 1
        if len(selected) >= max_records:
            break
    return selected


def consolidate_records_with_quota(
    primary: Iterable[Path],
    fallback: Iterable[Path],
    *,
    max_records: int,
    max_per_patient: int,
    positive_quota: int | None = None,
    negative_quota: int | None = None,
) -> list[Path]:
    if max_records <= 0:
        return []

    primary_list = list(primary)
    fallback_list = list(fallback)

    pos_candidates: list[Path] = []
    neg_candidates: list[Path] = []
    for candidate in primary_list + fallback_list:
        if candidate is None:
            continue
        path = Path(candidate)
        if not path.exists():
            continue
        csvp = path.with_name(path.stem + "_bi.csv")
        intervals = read_intervals(csvp)
        if intervals:
            pos_candidates.append(path)
        else:
            neg_candidates.append(path)

    selected: list[Path] = []
    seen: set[str] = set()
    patient_counts: dict[str, int] = defaultdict(int)

    def _take_from(source: list[Path], limit: int | None):
        nonlocal selected
        if limit is None or limit <= 0:
            return
        for p in source:
            if len(selected) >= max_records:
                break
            key = str(Path(p).resolve())
            if key in seen:
                continue
            pid = p.stem.split("_")[0]
            if max_per_patient > 0 and patient_counts[pid] >= max_per_patient:
                continue
            selected.append(p)
            seen.add(key)
            patient_counts[pid] += 1
            if limit is not None and len([s for s in selected if s in source]) >= limit:
                break

    primary_pos = [p for p in primary_list if p in pos_candidates]
    primary_neg = [p for p in primary_list if p in neg_candidates]

    _take_from(primary_pos, positive_quota)
    _take_from(primary_neg, negative_quota)

    remaining_slots = max_records - len(selected)
    if remaining_slots > 0:
        for p in primary_list:
            if len(selected) >= max_records:
                break
            key = str(Path(p).resolve())
            if key in seen:
                continue
            pid = p.stem.split("_")[0]
            if max_per_patient > 0 and patient_counts[pid] >= max_per_patient:
                continue
            selected.append(p)
            seen.add(key)
            patient_counts[pid] += 1

    if len(selected) < max_records:
        for p in pos_candidates + neg_candidates:
            if len(selected) >= max_records:
                break
            key = str(Path(p).resolve())
            if key in seen:
                continue
            pid = p.stem.split("_")[0]
            if max_per_patient > 0 and patient_counts[pid] >= max_per_patient:
                continue
            selected.append(p)
            seen.add(key)
            patient_counts[pid] += 1

    return selected


def build_mask(n_samples: int, intervals_sec: list[tuple[float, float]], fs: float) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=np.int8)
    for start_sec, stop_sec in intervals_sec:
        s = max(0, int(np.floor(start_sec * fs)))
        e = min(n_samples, int(np.ceil(stop_sec * fs)))
        if e > s:
            mask[s:e] = 1
    return mask


def compute_spectral_features(eeg_tc: np.ndarray, fs: float, eps: float) -> dict[str, float]:
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
    p_norm = total_band / (total_power[None, :] + eps)
    ent = -np.sum(p_norm * np.log(p_norm + eps), axis=0)
    K = total_band.shape[0]
    spectral_entropy = float(np.mean(ent / (np.log(K + eps))))

    csum = np.cumsum(total_band, axis=0)
    thresholds = 0.95 * (total_power + eps)
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
        "bp_rel_delta": bp_delta / (total_scalar + eps),
        "bp_rel_theta": bp_theta / (total_scalar + eps),
        "bp_rel_alpha": bp_alpha / (total_scalar + eps),
        "bp_rel_beta": bp_beta / (total_scalar + eps),
        "bp_rel_gamma": bp_gamma / (total_scalar + eps),
        "spectral_entropy": spectral_entropy,
        "sef95": sef95,
    }


def compute_feature_vector(eeg_tc: np.ndarray, fs: float, *, eps: float) -> dict[str, float]:
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
        mobility = np.sqrt((var1 + eps) / (var0 + eps))
        ddx = np.diff(dx, axis=0)
        var2 = np.mean(ddx ** 2, axis=0)
        complexity = np.sqrt((var2 + eps) / (var1 + eps)) / (mobility + eps)
        feats["hjorth_activity"] = float(np.mean(var0))
        feats["hjorth_mobility"] = float(np.mean(mobility))
        feats["hjorth_complexity"] = float(np.mean(complexity))
    else:
        feats["hjorth_activity"] = 0.0
        feats["hjorth_mobility"] = 0.0
        feats["hjorth_complexity"] = 0.0

    spec = compute_spectral_features(x, fs, eps)
    feats.update(spec)

    bp_alpha = spec["bp_alpha"] + eps
    bp_beta = spec["bp_beta"] + eps
    bp_theta = spec["bp_theta"] + eps

    feats["beta_alpha_ratio"] = spec["bp_beta"] / bp_alpha
    feats["theta_alpha_ratio"] = spec["bp_theta"] / bp_alpha
    feats["ratio_theta_alpha_over_beta"] = (spec["bp_theta"] + spec["bp_alpha"]) / bp_beta
    feats["ratio_theta_beta"] = spec["bp_theta"] / bp_beta
    feats["ratio_theta_alpha_over_alpha_beta"] = (
        (spec["bp_theta"] + spec["bp_alpha"]) / (spec["bp_alpha"] + spec["bp_beta"] + eps)
    )
    return feats


def _feature_row_worker(args: tuple[np.ndarray, float, float, Sequence[str]]) -> list[float]:
    window, fs, eps, feature_names = args
    feats = compute_feature_vector(window, fs, eps=eps)
    return [float(feats[name]) for name in feature_names]


@dataclass
class DatasetBundle:
    sequences: np.ndarray
    features: np.ndarray | None
    feature_names: list[str]
    labels: np.ndarray
    patients: np.ndarray
    records: np.ndarray
    label_mode: str = "window"
    storage: str = "ram"
    artifacts: dict[str, object] = field(default_factory=dict)


class _MemmapAccumulator:
    def __init__(
        self,
        *,
        base_dir: Path,
        prefix: str,
        window_shape: tuple[int, int],
        label_shape: tuple[int, ...],
        label_dtype: np.dtype,
        include_features: bool,
        feature_dim: int | None,
        sequence_dtype: np.dtype = np.float32,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.window_shape = tuple(int(x) for x in window_shape)
        self.label_shape = tuple(int(x) for x in label_shape)
        self.sequence_dtype = np.dtype(sequence_dtype)
        self.label_dtype = np.dtype(label_dtype)
        self.include_features = include_features
        self.feature_dim = feature_dim

        self._sequence_path = self.base_dir / f"{self.prefix}_seq.dat"
        self._label_path = self.base_dir / f"{self.prefix}_labels.dat"
        self._feature_path = (
            self.base_dir / f"{self.prefix}_feat.dat" if include_features else None
        )

        self._sequence_mm: np.memmap | None = None
        self._label_mm: np.memmap | None = None
        self._feature_mm: np.memmap | None = None
        self._capacity = 0
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def artifacts(self) -> dict[str, object]:
        info: dict[str, object] = {
            "storage": "memmap",
            "sequence_path": str(self._sequence_path),
            "label_path": str(self._label_path),
        }
        if self._feature_path is not None:
            info["feature_path"] = str(self._feature_path)
        info["window_shape"] = self.window_shape
        info["label_shape"] = self.label_shape
        info["sequence_dtype"] = str(self.sequence_dtype)
        info["label_dtype"] = str(self.label_dtype)
        if self.feature_dim is not None:
            info["feature_dim"] = int(self.feature_dim)
        return info

    def _ensure_capacity(self, required: int) -> None:
        if required <= self._capacity:
            return
        new_capacity = max(required, self._capacity * 2 if self._capacity else 1024)
        self._resize_sequences(new_capacity)
        self._resize_labels(new_capacity)
        if self.include_features and self.feature_dim is not None:
            self._resize_features(new_capacity)
        self._capacity = new_capacity

    def _resize_sequences(self, capacity: int) -> None:
        bytes_per_sample = int(np.prod(self.window_shape)) * self.sequence_dtype.itemsize
        total_bytes = capacity * bytes_per_sample
        if self._sequence_mm is not None:
            self._sequence_mm.flush()
            del self._sequence_mm
        mode = "r+b" if self._sequence_path.exists() else "w+b"
        with open(self._sequence_path, mode) as fp:
            fp.truncate(total_bytes)
        self._sequence_mm = np.memmap(
            self._sequence_path,
            dtype=self.sequence_dtype,
            mode="r+",
            shape=(capacity, *self.window_shape),
        )

    def _resize_labels(self, capacity: int) -> None:
        label_shape = (capacity, *self.label_shape) if self.label_shape else (capacity,)
        bytes_per_sample = int(np.prod(label_shape[1:]) or 1) * self.label_dtype.itemsize
        total_bytes = capacity * bytes_per_sample
        if self._label_mm is not None:
            self._label_mm.flush()
            del self._label_mm
        mode = "r+b" if self._label_path.exists() else "w+b"
        with open(self._label_path, mode) as fp:
            fp.truncate(total_bytes)
        self._label_mm = np.memmap(
            self._label_path,
            dtype=self.label_dtype,
            mode="r+",
            shape=label_shape,
        )

    def _resize_features(self, capacity: int) -> None:
        if self._feature_path is None or self.feature_dim is None:
            return
        bytes_per_sample = self.feature_dim * np.dtype(np.float32).itemsize
        total_bytes = capacity * bytes_per_sample
        if self._feature_mm is not None:
            self._feature_mm.flush()
            del self._feature_mm
        mode = "r+b" if self._feature_path.exists() else "w+b"
        with open(self._feature_path, mode) as fp:
            fp.truncate(total_bytes)
        self._feature_mm = np.memmap(
            self._feature_path,
            dtype=np.float32,
            mode="r+",
            shape=(capacity, self.feature_dim),
        )

    def ensure_feature_dim(self, dim: int | None) -> None:
        if not self.include_features:
            return
        if dim is None:
            raise RuntimeError("Dimensión de features desconocida para memmap.")
        if self.feature_dim is None:
            self.feature_dim = int(dim)
            if self._capacity:
                self._resize_features(self._capacity)
        elif self.feature_dim != int(dim):
            raise RuntimeError(
                f"Dimensión de features inconsistente: se esperaba {self.feature_dim}, se recibió {dim}."
            )

    def append(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        features: np.ndarray | None,
    ) -> None:
        sequences = np.asarray(sequences, dtype=self.sequence_dtype)
        labels = np.asarray(labels, dtype=self.label_dtype)
        if sequences.ndim != 3:
            raise ValueError("Las secuencias memmap deben tener forma (N, T, C).")
        batch = int(sequences.shape[0])
        if labels.shape[0] != batch:
            raise ValueError("El número de etiquetas no coincide con las secuencias.")
        if self.include_features:
            if features is None:
                raise RuntimeError("Se solicitaron features pero no se proporcionaron datos.")
            features = np.asarray(features, dtype=np.float32)
            if features.shape[0] != batch:
                raise ValueError("El número de features no coincide con las secuencias.")
            self.ensure_feature_dim(int(features.shape[1]))

        if self._sequence_mm is None:
            self._ensure_capacity(batch)
        else:
            self._ensure_capacity(self._count + batch)

        if self._sequence_mm is None or self._label_mm is None:
            raise RuntimeError("Memmap interno no inicializado correctamente.")

        start = self._count
        end = start + batch
        self._sequence_mm[start:end] = sequences
        self._label_mm[start:end] = labels
        if self.include_features and self._feature_mm is not None:
            self._feature_mm[start:end] = features
        self._count = end

    def finalize(self) -> tuple[np.memmap, np.memmap, np.memmap | None]:
        if self._sequence_mm is None or self._label_mm is None:
            raise RuntimeError("Memmap interno no inicializado.")
        self._sequence_mm.flush()
        self._label_mm.flush()
        if self._feature_mm is not None:
            self._feature_mm.flush()
        if self._capacity != self._count:
            seq_bytes = self._count * int(np.prod(self.window_shape)) * self.sequence_dtype.itemsize
            with open(self._sequence_path, "r+b") as fp:
                fp.truncate(seq_bytes)
            label_inner = int(np.prod(self.label_shape) or 1)
            label_bytes = self._count * label_inner * self.label_dtype.itemsize
            with open(self._label_path, "r+b") as fp:
                fp.truncate(label_bytes)
            if self.include_features and self._feature_path is not None and self.feature_dim is not None:
                feat_bytes = self._count * self.feature_dim * np.dtype(np.float32).itemsize
                with open(self._feature_path, "r+b") as fp:
                    fp.truncate(feat_bytes)
        seq_mm = np.memmap(
            self._sequence_path,
            dtype=self.sequence_dtype,
            mode="r+",
            shape=(self._count, *self.window_shape),
        )
        label_shape = (self._count, *self.label_shape) if self.label_shape else (self._count,)
        lbl_mm = np.memmap(
            self._label_path,
            dtype=self.label_dtype,
            mode="r+",
            shape=label_shape,
        )
        feat_mm = None
        if self.include_features and self._feature_path is not None and self.feature_dim is not None:
            feat_mm = np.memmap(
                self._feature_path,
                dtype=np.float32,
                mode="r+",
                shape=(self._count, self.feature_dim),
            )
        self._sequence_mm = seq_mm
        self._label_mm = lbl_mm
        self._feature_mm = feat_mm
        self._capacity = self._count
        return seq_mm, lbl_mm, feat_mm


def _flatten_labels_to_frames(labels: np.ndarray, *, label_mode: str) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.size == 0:
        return arr.reshape(0, 1)
    if label_mode == "time_step" and arr.ndim >= 2:
        return arr.reshape(arr.shape[0], -1)
    return arr.reshape(arr.shape[0], 1)


def _window_level_labels(labels: np.ndarray, *, label_mode: str) -> np.ndarray:
    frames = _flatten_labels_to_frames(labels, label_mode=label_mode)
    if frames.size == 0:
        return np.zeros((frames.shape[0],), dtype=np.int32)
    return (frames.max(axis=1) >= 0.5).astype(np.int32)


def _positive_frame_ratio(labels: np.ndarray, *, label_mode: str) -> float:
    frames = _flatten_labels_to_frames(labels, label_mode=label_mode)
    if frames.size == 0:
        return 0.0
    return float(frames.mean())


def _ensure_probability_matrix(prob_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(prob_array)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim >= 3:
        return arr.reshape(arr.shape[0], -1)
    return arr


def count_seizure_windows(labels: Sequence[int] | np.ndarray, *, label_mode: str = "window") -> int:
    arr = np.asarray(labels)
    if arr.size == 0:
        return 0
    if label_mode == "time_step" and arr.ndim >= 2:
        flags = _window_level_labels(arr, label_mode=label_mode)
        return int(flags.sum())
    return int(np.sum(arr.astype(np.int32)))


def summarize_dataset_bundle(name: str, bundle: DatasetBundle) -> dict[str, object]:
    label_mode = getattr(bundle, "label_mode", "window")
    frame_matrix = _flatten_labels_to_frames(bundle.labels, label_mode=label_mode)
    total_windows = int(frame_matrix.shape[0])
    total_frames = int(frame_matrix.size)
    frame_positive = int(frame_matrix.sum())
    frame_ratio = float(frame_matrix.mean()) if total_frames else 0.0

    window_flags = _window_level_labels(bundle.labels, label_mode=label_mode)
    seizure_windows = int(window_flags.sum())
    window_ratio = float(window_flags.mean()) if total_windows else 0.0

    unique_patients = int(np.unique(bundle.patients).size)
    unique_records = int(np.unique(bundle.records).size)

    record_labels: dict[str, list[int]] = defaultdict(list)
    for rec_id, flag in zip(bundle.records, window_flags):
        record_labels[str(rec_id)].append(int(flag))
    records_with_seizure = sum(1 for labels in record_labels.values() if any(lbl > 0 for lbl in labels))
    background_only_records = unique_records - records_with_seizure

    summary = {
        "name": name,
        "windows": total_windows,
        "frames": total_frames,
        "seizure_windows": seizure_windows,
        "positive_frames": frame_positive,
        "unique_patients": unique_patients,
        "unique_records": unique_records,
        "records_with_seizure": records_with_seizure,
        "records_background_only": background_only_records,
        "positive_ratio": window_ratio,
        "frame_positive_ratio": frame_ratio,
    }

    if label_mode == "time_step":
        print(
            f"📊 Resumen {name}: ventanas={total_windows} (positividad ventanas={window_ratio:.3f}), "
            f"frames={total_frames} (positividad frames={frame_ratio:.3f}), registros con convulsión={records_with_seizure}, "
            f"solo fondo={background_only_records}, pacientes únicos={unique_patients}, registros únicos={unique_records}"
        )
    else:
        print(
            f"📊 Resumen {name}: ventanas={total_windows}, convulsiones={seizure_windows}, "
            f"registros con convulsión={records_with_seizure}, solo fondo={background_only_records}, "
            f"pacientes únicos={unique_patients}, registros únicos={unique_records}, "
            f"positividad={window_ratio:.3f}"
        )

    return summary


def _feature_signature_token(signature: object | None) -> str | None:
    if signature is None:
        return None
    if isinstance(signature, (list, tuple)):
        flattened = ",".join(str(item) for item in signature)
    elif isinstance(signature, set):
        flattened = ",".join(sorted(str(item) for item in signature))
    else:
        flattened = str(signature)
    digest = hashlib.sha1(flattened.encode("utf-8")).hexdigest()[:8]
    return f"feat-{digest}"


def resolve_cache_target(
    cache_option: str | bool | None,
    suffix: str | None = None,
    *,
    label_mode: str | None = None,
    feature_signature: object | None = None,
) -> str | bool | None:
    if cache_option is None or cache_option is False:
        return None
    if isinstance(cache_option, str):
        suffix_tokens: list[str] = []
        if suffix:
            suffix_tokens.append(suffix)
        if label_mode:
            suffix_tokens.append(f"lm-{label_mode}")
        feat_token = _feature_signature_token(feature_signature)
        if feat_token:
            suffix_tokens.append(feat_token)
        suffix_part = f"_{'_'.join(suffix_tokens)}" if suffix_tokens else ""
        return f"{cache_option}{suffix_part}"
    return True


CACHE_MAGIC = "TPTCACHE_V1"
CACHE_VERSION = 1


def _cache_file_path(base_dir: Path, kind: str, split_name: str, entity_id: str) -> Path:
    return Path(base_dir) / f"by_{kind}" / split_name / f"{entity_id}.npz"


def _write_np_cache(
    path: Path,
    *,
    sequences: np.ndarray,
    labels: np.ndarray,
    patients: Sequence[str],
    records: Sequence[str],
    label_mode: str,
    hop_samples: int | None,
    target_fs: float | None,
    features: np.ndarray | None = None,
    feature_names: Sequence[str] | None = None,
    compression: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    seq_array = np.asarray(sequences, dtype=np.float32)
    label_array = np.asarray(labels)
    if label_mode == "window":
        label_array = label_array.astype(np.int32, copy=False)
    else:
        label_array = label_array.astype(np.float32, copy=False)

    payload: dict[str, np.ndarray] = {
        "magic": np.array([CACHE_MAGIC], dtype="<U16"),
        "version": np.array([CACHE_VERSION], dtype=np.int32),
        "label_mode": np.array([label_mode], dtype="<U16"),
        "hop_samples": np.array([hop_samples if hop_samples is not None else -1], dtype=np.int64),
        "target_fs": np.array(
            [target_fs if target_fs is not None else np.nan],
            dtype=np.float32,
        ),
        "sequences": seq_array,
        "labels": label_array,
        "patients": np.asarray(list(patients), dtype="<U64"),
        "records": np.asarray(list(records), dtype="<U128"),
    }
    if features is not None:
        payload["features"] = np.asarray(features, dtype=np.float32)
    if feature_names is not None:
        payload["feature_names"] = np.asarray(list(feature_names), dtype="<U128")

    saver = np.savez_compressed if compression else np.savez
    saver(path, **payload)


def _load_np_cache(
    path: Path,
    *,
    include_features: bool,
    expected_length: int | None,
    expected_hop_samples: int | None,
    expected_target_fs: float | None,
    label_mode_expected: str,
    entity_name: str,
) -> dict[str, object] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            magic = str(data.get("magic", np.array([""], dtype="<U1"))[0])
            if magic != CACHE_MAGIC:
                print(f"⚠️  {entity_name}: cache incompatible (formato inesperado).")
                return None
            label_mode_stored = str(data.get("label_mode", np.array(["window"], dtype="<U16"))[0])
            if label_mode_stored != label_mode_expected:
                print(
                    f"⚠️  {entity_name}: modo de etiqueta '{label_mode_stored}' incompatible con la configuración actual."
                )
                return None

            sequences = np.asarray(data["sequences"], dtype=np.float32)
            labels = np.asarray(data["labels"])
            if expected_length is not None and sequences.shape[1] != expected_length:
                print(
                    f"⚠️  {entity_name}: caché descartada por longitud de ventana {sequences.shape[1]} != {expected_length}."
                )
                return None

            hop_arr = data.get("hop_samples")
            hop_value = int(hop_arr[0]) if hop_arr is not None else -1
            hop_samples = None if hop_value < 0 else hop_value
            if (
                expected_hop_samples is not None
                and hop_samples is not None
                and hop_samples != expected_hop_samples
            ):
                print(
                    f"⚠️  {entity_name}: caché descartada por hop_samples {hop_samples} != {expected_hop_samples}."
                )
                return None

            target_arr = data.get("target_fs")
            target_value = float(target_arr[0]) if target_arr is not None else float("nan")
            target_fs = None if math.isnan(target_value) else target_value
            if (
                expected_target_fs is not None
                and target_fs is not None
                and not math.isclose(expected_target_fs, target_fs, rel_tol=1e-6, abs_tol=1e-3)
            ):
                print(
                    f"⚠️  {entity_name}: caché descartada por frecuencia objetivo {target_fs:.6g} != {expected_target_fs:.6g}."
                )
                return None

            patients = data.get("patients")
            records = data.get("records")
            if patients is None or records is None:
                print(f"⚠️  {entity_name}: caché inválida (faltan IDs de paciente/registro).")
                return None
            patient_ids = np.asarray(patients, dtype=str)
            record_ids = np.asarray(records, dtype=str)

            features = None
            feature_names = None
            if include_features:
                if "features" not in data:
                    print(f"⚠️  {entity_name}: caché sin features requeridas; se regenerará.")
                    return None
                features = np.asarray(data["features"], dtype=np.float32)
                if "feature_names" in data:
                    feature_names = np.asarray(data["feature_names"], dtype=str).tolist()

    except Exception as exc:  # pylint: disable=broad-except
        print(f"⚠️  {entity_name}: error al leer la caché ({exc}).")
        return None

    return {
        "sequences": sequences,
        "labels": labels,
        "patients": patient_ids,
        "records": record_ids,
        "features": features,
        "feature_names": feature_names,
        "hop_samples": hop_samples,
        "target_fs": target_fs,
    }


def prepare_model_inputs(
    sequence_array: np.ndarray,
    feature_array: np.ndarray | None,
    *,
    as_torch: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
):
    if feature_array is None:
        result = sequence_array
    else:
        result = (sequence_array, feature_array)

    if not as_torch:
        return result

    target_dtype = dtype or torch.float32
    if feature_array is None:
        tensor = torch.as_tensor(sequence_array, dtype=target_dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    seq_tensor = torch.as_tensor(sequence_array, dtype=target_dtype)
    feat_tensor = torch.as_tensor(feature_array, dtype=target_dtype)
    if device is not None:
        seq_tensor = seq_tensor.to(device)
        feat_tensor = feat_tensor.to(device)
    return seq_tensor, feat_tensor


class WindowDataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        *,
        features: np.ndarray | None = None,
        label_mode: str = "window",
    transform: Callable | None = None,
    ) -> None:
        self.sequences = sequences
        self.labels = labels
        self.features = features
        self.label_mode = label_mode
        self.transform = transform

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        seq = torch.as_tensor(np.asarray(self.sequences[idx]), dtype=torch.float32)
        lbl_array = np.asarray(self.labels[idx])
        if self.label_mode == "time_step":
            lbl = torch.as_tensor(lbl_array, dtype=torch.float32)
        else:
            lbl = torch.as_tensor(float(lbl_array), dtype=torch.float32)

        if self.features is not None:
            feat = torch.as_tensor(np.asarray(self.features[idx]), dtype=torch.float32)
            sample = (seq, feat)
        else:
            sample = seq

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, lbl


def build_torch_dataset_from_arrays(
    inputs: np.ndarray | tuple[np.ndarray, np.ndarray],
    labels: np.ndarray,
    *,
    label_mode: str = "window",
    transform: Callable | None = None,
) -> WindowDataset:
    if isinstance(inputs, tuple):
        seq_array, feat_array = inputs
    else:
        seq_array = inputs
        feat_array = None
    return WindowDataset(
        seq_array,
        labels,
        features=feat_array,
        label_mode=label_mode,
        transform=transform,
    )


def build_dataloader(
    bundle: DatasetBundle,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
    transform: Callable | None = None,
) -> DataLoader:
    dataset = WindowDataset(
        bundle.sequences,
        bundle.labels,
        features=bundle.features,
        label_mode=bundle.label_mode,
        transform=transform,
    )
    persistent = bool(num_workers) and bool(persistent_workers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )


def _normalize_montage_name(name: str | None) -> str | None:
    if name is None:
        return None
    normalized = name.strip().lower().replace("-", "_")
    return normalized or None


def _matches_montage_filter(path: Path, montage_name: str | None) -> bool:
    if montage_name is None:
        return True
    montage_key = _normalize_montage_name(montage_name)
    if montage_key is None:
        return True
    path_str = str(path).lower()
    if montage_key == "ar":
        return "_tcp_ar" in path_str and "_tcp_ar_a" not in path_str
    if montage_key in {"ar_a", "ara"}:
        return "_tcp_ar_a" in path_str
    if montage_key == "le":
        return "_tcp_le" in path_str
    return True


def discover_records(
    base_dir: Path,
    *,
    max_records: int,
    max_per_patient: int,
    include_background_only: bool = False,
    montage: str | None = None,
) -> list[Path]:
    if not base_dir.exists():
        return []
    csv_files = [path for path in sorted(base_dir.rglob("*_bi.csv")) if _matches_montage_filter(path, montage)]
    if max_records <= 0:
        return []

    selected: list[Path] = []
    seen: set[str] = set()
    patient_counts: defaultdict[str, int] = defaultdict(int)

    for csv_path in csv_files:
        intervals = read_intervals(csv_path)
        has_seizure = bool(intervals)
        if not has_seizure and not include_background_only:
            continue

        stem = csv_path.stem
        candidates = [csv_path.with_suffix(".edf")]
        if stem.endswith("_bi"):
            candidates.append(csv_path.with_name(stem[:-3] + ".edf"))
        candidates.append(csv_path.with_suffix(".edf.gz"))
        if stem.endswith("_bi"):
            candidates.append(csv_path.with_name(stem[:-3] + ".edf.gz"))

        valid_candidates = [p for p in candidates if p.exists() and _matches_montage_filter(p, montage)]
        edf_path = next(iter(valid_candidates), None)
        if edf_path is None:
            continue

        key = str(edf_path.resolve())
        if key in seen:
            continue

        patient_id = edf_path.stem.split("_")[0]
        if max_per_patient > 0 and patient_counts[patient_id] >= max_per_patient:
            continue

        selected.append(edf_path)
        seen.add(key)
        patient_counts[patient_id] += 1

        if len(selected) >= max_records:
            break

    return selected


def discover_records_multi(
    base_dirs: Iterable[Path],
    *,
    max_records: int,
    max_per_patient: int,
    include_background_only: bool = False,
    montage: str | None = None,
) -> tuple[list[Path], dict[str, int]]:
    collected: list[Path] = []
    seen: set[str] = set()
    patient_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for base_dir in base_dirs:
        if len(collected) >= max_records:
            break
        remaining = max_records - len(collected)
        for path in discover_records(
            base_dir,
            max_records=remaining,
            max_per_patient=max_per_patient,
            include_background_only=include_background_only,
            montage=montage,
        ):
            key = str(path.resolve())
            if key in seen:
                continue
            patient_id = path.stem.split("_")[0]
            patient_counts.setdefault(patient_id, 0)
            if patient_counts[patient_id] >= max_per_patient:
                continue
            split_name = "unknown"
            parts = path.parts
            if "edf" in parts:
                idx = parts.index("edf")
                if idx + 1 < len(parts):
                    split_name = parts[idx + 1]
            collected.append(path)
            seen.add(key)
            patient_counts[patient_id] += 1
            split_counts[split_name] = split_counts.get(split_name, 0) + 1
            if len(collected) >= max_records:
                break
    return collected, split_counts


def build_windows_dataset(
    records: Iterable[Path],
    *,
    condition: str,
    montage: str,
    include_features: bool,
    time_step_labels: bool,
    feature_subset: Sequence[str] | None,
    min_positive_ratio: float = 0.0,
    window_sec: float,
    hop_sec: float,
    eps: float,
    target_fs: float,
    preprocess_settings: dict[str, object],
    include_background_only: bool = False,
    sampling_strategy: str = "none",
    sampling_seed: int | None = None,
    target_positive_ratio: float | None = None,
    target_positive_ratio_tolerance: float = 0.0,
    undersample_seed: int | None = None,
    tfrecord_export=None,
    storage_mode: str = "ram",
    memmap_dir: Path | str | None = None,
    memmap_prefix: str | None = None,
    auto_memmap_threshold_bytes: int | None = None,
    feature_worker_processes: int | None = None,
    feature_worker_chunk_size: int | None = None,
    feature_parallel_min_windows: int | None = None,
    force_memmap_after_build: bool = False,
) -> DatasetBundle:
    config = dict(preprocess_settings)
    expected_length = int(round(window_sec * target_fs)) if target_fs > 0 and window_sec > 0 else None
    expected_hop_samples = int(round(hop_sec * target_fs)) if target_fs > 0 and hop_sec > 0 else None

    unique_records: list[Path] = []
    seen_paths: set[str] = set()
    for candidate in records:
        resolved = str(Path(candidate).resolve())
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        unique_records.append(Path(candidate))

    storage_mode_normalized = (storage_mode or "ram").strip().lower()
    auto_mode = storage_mode_normalized == "auto"
    use_memmap = storage_mode_normalized == "memmap"
    if storage_mode_normalized not in {"ram", "memmap", "auto"}:
        raise ValueError("'storage_mode' debe ser 'ram', 'memmap' o 'auto'.")

    sampling_strategy_lower = (sampling_strategy or "none").lower()
    if use_memmap and sampling_strategy_lower != "none":
        raise ValueError(
            "Las estrategias de muestreo distintas de 'none' no están disponibles con storage_mode='memmap'."
        )
    if use_memmap and target_positive_ratio is not None:
        raise ValueError(
            "El undersampling por ratio objetivo no está disponible con storage_mode='memmap'."
        )

    if use_memmap or auto_mode:
        if memmap_dir is None:
            memmap_root = Path(tempfile.gettempdir()) / "tfdev_memmap"
        else:
            memmap_root = Path(memmap_dir).expanduser().resolve()
        memmap_root.mkdir(parents=True, exist_ok=True)
        memmap_prefix_value = memmap_prefix or f"{condition}_{uuid4().hex[:8]}"
    else:
        memmap_root = None
        memmap_prefix_value = None

    sequence_batches: list[np.ndarray] = []
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    patient_list: list[str] = []
    record_list: list[str] = []
    feature_keys: list[str] | None = None
    requested_features = [str(name) for name in feature_subset] if feature_subset else None

    write_enabled = bool(tfrecord_export and getattr(tfrecord_export, "write_enabled", False))
    reuse_enabled = bool(tfrecord_export and getattr(tfrecord_export, "reuse_enabled", False))
    compression = getattr(tfrecord_export, "compression", None) if tfrecord_export else None
    cache_base_dir: Path | None = None
    cache_split = ""
    if tfrecord_export is not None:
        cache_base_dir = Path(tfrecord_export.base_dir).expanduser().resolve()
        cache_split = tfrecord_export.split_name

    cache_enabled = write_enabled or reuse_enabled
    if cache_enabled:
        if cache_base_dir is None:
            raise ValueError("Se habilitó la caché pero no se proporcionó 'base_dir'.")
        cache_base_dir.mkdir(parents=True, exist_ok=True)

    if time_step_labels and cache_enabled:
        raise RuntimeError(
            "La caché NPZ actual no soporta 'time_step_labels'. Desactívala o usa salidas por ventana."
        )

    use_npz_compression = bool(compression)
    cache_split_name = cache_split or "unspecified"
    label_mode_expected = "time_step" if time_step_labels else "window"
    expected_target_for_cache = target_fs if target_fs > 0 else None

    patient_acc: defaultdict[str, dict[str, list]] | None = None
    if write_enabled:
        patient_acc = defaultdict(
            lambda: {
                "sequences": [],
                "labels": [],
                "records": [],
                "features": [],
                "hop_samples": None,
                "target_fs": None,
            }
        )
    patients_needing_export: set[str] = set()

    memmap_builder: _MemmapAccumulator | None = None
    memmap_artifacts: dict[str, object] | None = None

    promote_threshold_bytes: int | None = None
    if auto_mode and auto_memmap_threshold_bytes is not None and auto_memmap_threshold_bytes > 0:
        promote_threshold_bytes = int(auto_memmap_threshold_bytes)
    ram_usage_bytes = 0

    parallel_workers = None
    if feature_worker_processes is not None and feature_worker_processes > 0:
        parallel_workers = int(feature_worker_processes)
    parallel_chunk_size = None
    if feature_worker_chunk_size is not None and feature_worker_chunk_size > 0:
        parallel_chunk_size = int(feature_worker_chunk_size)
    min_windows_parallel = 32
    if feature_parallel_min_windows is not None and feature_parallel_min_windows > 0:
        min_windows_parallel = int(feature_parallel_min_windows)

    def _ensure_memmap_initialized(
        window_shape: tuple[int, int],
        label_shape: tuple[int, ...],
        label_dtype: np.dtype,
        feature_dim: int | None,
    ) -> _MemmapAccumulator:
        nonlocal memmap_builder, memmap_artifacts
        if memmap_root is None or memmap_prefix_value is None:
            raise RuntimeError("Configuración inválida de memmap (ruta o prefijo no definidos).")
        if memmap_builder is None:
            memmap_builder = _MemmapAccumulator(
                base_dir=memmap_root,
                prefix=memmap_prefix_value,
                window_shape=window_shape,
                label_shape=label_shape,
                label_dtype=label_dtype,
                include_features=include_features,
                feature_dim=feature_dim,
            )
            memmap_artifacts = dict(memmap_builder.artifacts)
        return memmap_builder

    def _move_batches_to_memmap(
        window_shape: tuple[int, int],
        label_shape: tuple[int, ...],
        label_dtype: np.dtype,
        feature_dim: int | None,
    ) -> _MemmapAccumulator:
        nonlocal sequence_batches, label_batches, feature_batches, use_memmap, ram_usage_bytes, memmap_artifacts
        if sampling_strategy_lower != "none":
            raise RuntimeError(
                "Las estrategias de muestreo distintas de 'none' no están disponibles cuando se requiere memmap."
            )
        builder = _ensure_memmap_initialized(window_shape, label_shape, label_dtype, feature_dim)
        for idx, seq_batch in enumerate(sequence_batches):
            label_batch = label_batches[idx]
            feat_batch = feature_batches[idx] if include_features else None
            builder.append(seq_batch, label_batch, feat_batch)
        sequence_batches.clear()
        label_batches.clear()
        if include_features:
            feature_batches.clear()
        use_memmap = True
        ram_usage_bytes = 0
        memmap_artifacts = dict(builder.artifacts)
        gc.collect()
        return builder

    def _append_batch(
        seq_array: np.ndarray,
        label_array: np.ndarray,
        feature_array_local: np.ndarray | None,
    ) -> None:
        nonlocal ram_usage_bytes, use_memmap, memmap_artifacts
        if include_features and feature_array_local is None:
            raise RuntimeError("Se solicitaron features pero no se proporcionaron datos.")
        batch_bytes = seq_array.nbytes + label_array.nbytes
        if include_features and feature_array_local is not None:
            batch_bytes += feature_array_local.nbytes
        window_shape_local = tuple(int(x) for x in seq_array.shape[1:])
        if label_array.ndim > 1:
            label_shape_local = tuple(int(x) for x in label_array.shape[1:])
        else:
            label_shape_local = ()
        label_dtype_local = np.dtype(label_array.dtype)
        feature_dim_local = int(feature_array_local.shape[1]) if (include_features and feature_array_local is not None) else None

        if auto_mode and not use_memmap and promote_threshold_bytes is not None:
            projected = ram_usage_bytes + batch_bytes
            if projected > promote_threshold_bytes:
                builder = _move_batches_to_memmap(
                    window_shape_local,
                    label_shape_local,
                    label_dtype_local,
                    feature_dim_local,
                )
                builder.append(seq_array, label_array, feature_array_local)
                memmap_artifacts = dict(builder.artifacts)
                return

        if use_memmap:
            builder = _ensure_memmap_initialized(
                window_shape_local,
                label_shape_local,
                label_dtype_local,
                feature_dim_local,
            )
            builder.append(seq_array, label_array, feature_array_local)
            memmap_artifacts = dict(builder.artifacts)
        else:
            sequence_batches.append(seq_array)
            label_batches.append(label_array)
            if include_features and feature_array_local is not None:
                feature_batches.append(feature_array_local)
            ram_usage_bytes += batch_bytes

    for edf_path in unique_records:
        csv_path = edf_path.with_name(edf_path.stem + "_bi.csv")
        intervals = read_intervals(csv_path)
        record_id = edf_path.stem
        patient_id = record_id.split("_")[0]

        record_path: Path | None = None
        patient_path: Path | None = None
        record_exists = False
        patient_exists = False
        if cache_enabled and cache_base_dir is not None:
            record_path = _cache_file_path(cache_base_dir, "record", cache_split_name, record_id)
            patient_path = _cache_file_path(cache_base_dir, "patient", cache_split_name, patient_id)
            record_exists = record_path.exists()
            patient_exists = patient_path.exists()
            if write_enabled and not patient_exists:
                patients_needing_export.add(patient_id)

        if reuse_enabled and record_exists and record_path is not None:
            loaded = _load_np_cache(
                record_path,
                include_features=include_features,
                expected_length=expected_length,
                expected_hop_samples=expected_hop_samples,
                expected_target_fs=expected_target_for_cache,
                label_mode_expected=label_mode_expected,
                entity_name=record_id,
            )
            if loaded is not None:
                reused_sequences = np.asarray(loaded["sequences"], dtype=np.float32)
                loaded_labels = np.asarray(loaded["labels"])
                if label_mode_expected == "time_step":
                    reused_labels = loaded_labels.astype(np.float32, copy=False)
                    label_dtype = np.float32
                else:
                    reused_labels = loaded_labels.astype(np.int32, copy=False)
                    label_dtype = np.int32

                reused_features: np.ndarray | None = None
                loaded_feature_names = loaded.get("feature_names")
                can_reuse = True
                if include_features:
                    loaded_features = loaded.get("features")
                    if loaded_features is None:
                        can_reuse = False
                    else:
                        reused_features = np.asarray(loaded_features, dtype=np.float32)
                        if requested_features is not None:
                            expected_names = list(requested_features)
                            if feature_keys is None:
                                feature_keys = list(expected_names)
                            if reused_features.shape[1] != len(expected_names):
                                print(
                                    f"⚠️  {record_id}: la caché contiene {reused_features.shape[1]} features y"
                                    f" se solicitaron {len(expected_names)}; se regenerará desde EDF."
                                )
                                can_reuse = False
                        else:
                            loaded_names_list = (
                                list(loaded_feature_names)
                                if loaded_feature_names is not None
                                else None
                            )
                            if feature_keys is None:
                                if loaded_names_list is not None and len(loaded_names_list) == reused_features.shape[1]:
                                    feature_keys = list(loaded_names_list)
                                else:
                                    feature_keys = [f"feature_{idx}" for idx in range(reused_features.shape[1])]
                            elif len(feature_keys) != reused_features.shape[1]:
                                print(
                                    f"⚠️  {record_id}: dimensión de features en caché {reused_features.shape[1]}"
                                    f" ≠ {len(feature_keys)}; se regenerará desde EDF."
                                )
                                can_reuse = False
                            elif (
                                loaded_names_list is not None
                                and feature_keys != loaded_names_list
                                and all(name.startswith("feature_") for name in feature_keys)
                                and len(feature_keys) == len(loaded_names_list)
                            ):
                                feature_keys = list(loaded_names_list)

                if can_reuse:
                    _append_batch(reused_sequences, reused_labels, reused_features)

                    patient_list.extend(np.asarray(loaded["patients"], dtype=str).tolist())
                    record_list.extend(np.asarray(loaded["records"], dtype=str).tolist())

                    if (
                        write_enabled
                        and patient_path is not None
                        and not patient_exists
                        and patient_acc is not None
                    ):
                        bundle = patient_acc[patient_id]
                        bundle["sequences"].append(reused_sequences)
                        bundle["labels"].append(reused_labels)
                        bundle["records"].extend(np.asarray(loaded["records"], dtype=str).tolist())
                        if include_features and reused_features is not None:
                            bundle["features"].append(reused_features)
                        cached_hop = loaded.get("hop_samples")
                        cached_fs = loaded.get("target_fs")
                        if bundle["hop_samples"] is None and cached_hop is not None:
                            bundle["hop_samples"] = int(cached_hop)
                        if bundle["target_fs"] is None and cached_fs is not None:
                            bundle["target_fs"] = float(cached_fs)

                    print(
                        f"✓ {record_id}: {reused_labels.shape[0]} ventanas cargadas desde caché NPZ"
                    )
                    continue
            else:
                record_exists = False
                patient_exists = False
                if write_enabled and patient_acc is not None:
                    patients_needing_export.add(patient_id)

        if not intervals:
            if not include_background_only:
                print(f"⚠️  {record_id}: sin intervalos de convulsión; omitido.")
                continue
            print(f"ℹ️  {record_id}: solo etiquetas de fondo; se incluirá con ventanas negativas.")

        raw = None
        try:
            with apply_preprocess_config(config, target_fs=target_fs):
                raw = extract_montage_signals(
                    str(edf_path), montage=montage, desired_fs=target_fs
                )
            data = raw.get_data()
            fs = float(raw.info['sfreq'])
        except RuntimeError as err:
            print(f"⚠️  {record_id}: no se pudo generar ventanas ({err}).")
            continue
        finally:
            if raw is not None:
                raw.close()

        window_samples = int(round(window_sec * fs))
        hop_samples = int(round(hop_sec * fs))
        if data.shape[1] < window_samples:
            print(f"⚠️  {record_id}: duración insuficiente para una ventana completa.")
            continue

        mask = build_mask(data.shape[1], intervals, fs)

        available = data.shape[1] - window_samples
        n_windows = 1 + available // hop_samples
        if n_windows <= 0:
            print(f"⚠️  {record_id}: sin ventanas válidas después del preprocesamiento.")
            continue

        data_contig = np.ascontiguousarray(data, dtype=np.float32)
        window_view = as_strided(
            data_contig,
            shape=(data_contig.shape[0], n_windows, window_samples),
            strides=(data_contig.strides[0], data_contig.strides[1] * hop_samples, data_contig.strides[1]),
            writeable=False,
        )
        record_sequences = np.transpose(window_view, (1, 2, 0))
        if not record_sequences.flags.c_contiguous:
            record_sequences = np.ascontiguousarray(record_sequences)

        mask_contig = np.ascontiguousarray(mask, dtype=np.float32)
        mask_view = as_strided(
            mask_contig,
            shape=(n_windows, window_samples),
            strides=(mask_contig.strides[0] * hop_samples, mask_contig.strides[0]),
            writeable=False,
        )

        if time_step_labels:
            record_labels_arr = np.ascontiguousarray(mask_view, dtype=np.float32)
            label_mode_local = "time_step"
        else:
            record_labels_arr = (mask_view.mean(axis=1) >= 0.5).astype(np.int32)
            label_mode_local = "window"

        record_features_arr: np.ndarray | None = None
        feature_names_local: list[str] | None = None
        if include_features:
            feature_rows_list: list[list[float]] = []
            feat_names_candidate = list(requested_features) if requested_features is not None else None

            first_feats = compute_feature_vector(record_sequences[0], fs, eps=eps)
            if feat_names_candidate is None:
                feature_names_local = sorted(first_feats.keys())
            else:
                missing = [name for name in feat_names_candidate if name not in first_feats]
                if missing:
                    raise ValueError(
                        "Las siguientes features solicitadas no están disponibles: "
                        + ", ".join(sorted(missing))
                    )
                feature_names_local = feat_names_candidate

            feature_rows_list.append([float(first_feats[name]) for name in feature_names_local])

            remaining_indices = range(1, n_windows)
            if parallel_workers and n_windows >= max(2, min_windows_parallel):
                windows_iter = (record_sequences[idx] for idx in remaining_indices)
                with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
                    mapped_rows = list(
                        executor.map(
                            _feature_row_worker,
                            ((window, fs, eps, feature_names_local) for window in windows_iter),
                            chunksize=parallel_chunk_size if parallel_chunk_size else 1,
                        )
                    )
                feature_rows_list.extend(mapped_rows)
            else:
                for idx in remaining_indices:
                    feats = compute_feature_vector(record_sequences[idx], fs, eps=eps)
                    missing = [name for name in feature_names_local if name not in feats]
                    if missing:
                        raise ValueError(
                            "Las siguientes features solicitadas no están disponibles: "
                            + ", ".join(sorted(missing))
                        )
                    feature_rows_list.append([float(feats[name]) for name in feature_names_local])

            record_features_arr = np.asarray(feature_rows_list, dtype=np.float32)
            if feature_keys is None:
                feature_keys = list(feature_names_local)
            else:
                if feature_keys != feature_names_local:
                    placeholder = all(name.startswith("feature_") for name in feature_keys)
                    if placeholder and len(feature_keys) == len(feature_names_local):
                        feature_keys = list(feature_names_local)
                    else:
                        raise RuntimeError(
                            "La lista de features calculadas difiere de la previamente registrada."
                        )

        _append_batch(record_sequences, record_labels_arr, record_features_arr)

        if label_mode_local == "time_step":
            record_window_flags = _window_level_labels(record_labels_arr, label_mode="time_step")
            frame_positive_ratio = float(record_labels_arr.mean()) if record_labels_arr.size else 0.0
        else:
            record_window_flags = record_labels_arr
            frame_positive_ratio = float(record_labels_arr.mean()) if record_labels_arr.size else 0.0
        window_positive_ratio = float(record_window_flags.mean()) if record_window_flags.size else 0.0

        patient_list.extend([patient_id] * record_sequences.shape[0])
        record_list.extend([record_id] * record_sequences.shape[0])

        if write_enabled and record_path is not None:
            record_patients = [patient_id] * record_sequences.shape[0]
            record_ids_per_window = [record_id] * record_sequences.shape[0]
            labels_for_cache = (
                record_labels_arr.astype(np.int32, copy=False)
                if label_mode_local == "window"
                else record_labels_arr.astype(np.float32, copy=False)
            )
            _write_np_cache(
                record_path,
                sequences=record_sequences,
                labels=labels_for_cache,
                patients=record_patients,
                records=record_ids_per_window,
                label_mode=label_mode_local,
                hop_samples=hop_samples,
                target_fs=fs,
                features=record_features_arr if include_features else None,
                feature_names=feature_names_local if include_features else None,
                compression=compression if use_npz_compression else None,
            )
            if record_exists:
                print(f"   ↳ Caché de registro actualizada: {record_path}")
            else:
                print(f"   ↳ Caché de registro guardada: {record_path}")

        if (
            write_enabled
            and patient_acc is not None
            and patient_id in patients_needing_export
        ):
            patient_bundle = patient_acc[patient_id]
            patient_bundle["sequences"].append(record_sequences)
            patient_bundle["labels"].append(record_labels_arr)
            patient_bundle["records"].extend([record_id] * record_sequences.shape[0])
            if include_features and record_features_arr is not None:
                patient_bundle["features"].append(record_features_arr)
            if patient_bundle["hop_samples"] is None:
                patient_bundle["hop_samples"] = hop_samples
            elif patient_bundle["hop_samples"] != hop_samples:
                print(
                    f"⚠️  {record_id}: hop_samples inconsistente entre registros del mismo paciente; se mantiene el valor inicial."
                )
            if patient_bundle["target_fs"] is None:
                patient_bundle["target_fs"] = float(fs)
            elif not math.isclose(patient_bundle["target_fs"], float(fs), rel_tol=1e-6, abs_tol=1e-6):
                print(
                    f"⚠️  {record_id}: frecuencia objetivo inconsistente entre registros del mismo paciente; se mantiene {patient_bundle['target_fs']:.6g}."
                )

        if label_mode_local == "time_step":
            print(
                f"✓ {record_id}: {record_sequences.shape[0]} ventanas generadas "
                f"(positividad ventanas {window_positive_ratio:.3f}, frames {frame_positive_ratio:.3f})"
            )
        else:
            print(
                f"✓ {record_id}: {record_sequences.shape[0]} ventanas generadas "
                f"(positividad {window_positive_ratio:.3f})"
            )

        del record_sequences, mask_view, data_contig, window_view
        del record_labels_arr
        if include_features and record_features_arr is not None:
            del record_features_arr
        del mask_contig
        gc.collect()

    if use_memmap:
        if memmap_builder is None or memmap_builder.count == 0:
            raise RuntimeError("No se generaron ventanas para los registros proporcionados.")
        sequences, labels, feature_array = memmap_builder.finalize()
        storage_mode_final = "memmap"
    else:
        if not sequence_batches or not label_batches:
            raise RuntimeError("No se generaron ventanas para los registros proporcionados.")
        sequences = (
            sequence_batches[0]
            if len(sequence_batches) == 1
            else np.concatenate(sequence_batches, axis=0)
        )
        labels = (
            label_batches[0]
            if len(label_batches) == 1
            else np.concatenate(label_batches, axis=0)
        )
        feature_array = None
        if include_features:
            if not feature_batches:
                raise RuntimeError("Se solicitó include_features pero no se generaron features.")
            feature_array = (
                feature_batches[0]
                if len(feature_batches) == 1
                else np.concatenate(feature_batches, axis=0)
            )
        storage_mode_final = "ram"

        if force_memmap_after_build:
            target_root = memmap_root
            if target_root is None:
                if memmap_dir is not None:
                    target_root = Path(memmap_dir).expanduser().resolve()
                else:
                    target_root = Path(tempfile.gettempdir()) / "tfdev_memmap"
            target_root.mkdir(parents=True, exist_ok=True)
            final_prefix = memmap_prefix_value or f"{condition}_{uuid4().hex[:8]}"
            label_shape_final = labels.shape[1:] if labels.ndim > 1 else ()
            label_dtype_final = np.dtype(labels.dtype)
            feature_dim_final = int(feature_array.shape[1]) if (include_features and feature_array is not None) else None
            spool_builder = _MemmapAccumulator(
                base_dir=target_root,
                prefix=f"{final_prefix}_flush",
                window_shape=sequences.shape[1:],
                label_shape=label_shape_final,
                label_dtype=label_dtype_final,
                include_features=include_features,
                feature_dim=feature_dim_final,
            )
            seq_source = sequences
            lbl_source = labels
            feat_source = feature_array if include_features else None
            spool_builder.append(seq_source, lbl_source, feat_source)
            sequences_mm, labels_mm, feature_mm = spool_builder.finalize()
            sequences = sequences_mm
            labels = labels_mm
            feature_array = feature_mm
            storage_mode_final = "memmap"
            memmap_artifacts = dict(spool_builder.artifacts)
            memmap_artifacts["spooled_from_ram"] = True
            sequence_batches.clear()
            label_batches.clear()
            if include_features:
                feature_batches.clear()
            del seq_source, lbl_source
            if feat_source is not None:
                del feat_source
            gc.collect()

    patients = np.asarray(patient_list)
    records_arr = np.asarray(record_list)

    label_mode = "time_step" if time_step_labels else "window"

    if include_features:
        if feature_array is None:
            raise RuntimeError("Se solicitó include_features pero no se generaron features.")
        if feature_keys is None and requested_features is not None:
            feature_keys = list(requested_features)
        feature_names = feature_keys or []
    else:
        feature_names = []

    window_flags = _window_level_labels(labels, label_mode=label_mode)

    sampling_applied = False
    sampling_info: dict[str, object] | None = None
    undersampling_applied = False
    undersampling_info: dict[str, object] | None = None
    if sampling_strategy_lower != "none":
        rng = np.random.default_rng(sampling_seed)
        pos_idx = np.flatnonzero(window_flags == 1)
        neg_idx = np.flatnonzero(window_flags == 0)
        if sampling_strategy_lower == "balanced":
            minority = min(pos_idx.size, neg_idx.size)
            if minority == 0:
                print(
                    "⚠️  Estratificación 'balanced' omitida por falta de ejemplos en ambas clases."
                )
            else:
                pos_sel = pos_idx if pos_idx.size <= minority else rng.choice(pos_idx, size=minority, replace=False)
                neg_sel = neg_idx if neg_idx.size <= minority else rng.choice(neg_idx, size=minority, replace=False)
                selected_idx = np.concatenate([pos_sel, neg_sel])
                rng.shuffle(selected_idx)
                sequences = sequences[selected_idx]
                labels = labels[selected_idx]
                patients = patients[selected_idx]
                records_arr = records_arr[selected_idx]
                if feature_array is not None:
                    feature_array = feature_array[selected_idx]
                window_flags = window_flags[selected_idx]
                sampling_applied = True
                pos_windows = int(window_flags.sum())
                sampling_info = {
                    "strategy": sampling_strategy_lower,
                    "windows": int(selected_idx.size),
                    "positive_windows": pos_windows,
                    "negative_windows": int(selected_idx.size - pos_windows),
                }
                print(
                    f"   ↳ Stratified sampling ('balanced'): ventanas={selected_idx.size}, "
                    f"positivos={pos_windows}, negativos={int(selected_idx.size - pos_windows)}"
                )
        else:
            print(
                f"⚠️  Estrategia de muestreo '{sampling_strategy_lower}' no soportada; se omite."
            )

    if target_positive_ratio is not None and window_flags.size > 0:
        rng_seed = undersample_seed if undersample_seed is not None else sampling_seed
        rng = np.random.default_rng(rng_seed)
        current_ratio = float(window_flags.mean())
        tolerance = float(target_positive_ratio_tolerance)
        target_low = max(0.0, target_positive_ratio - tolerance)
        target_high = min(1.0, target_positive_ratio + tolerance)
        pos_idx = np.flatnonzero(window_flags == 1)
        neg_idx = np.flatnonzero(window_flags == 0)
        total_before = int(window_flags.size)
        selected_indices: np.ndarray | None = None
        drop_side: str | None = None
        achieved_ratio = current_ratio
        within_initial = target_low <= current_ratio <= target_high

        if within_initial:
            print(
                f"   ↳ Objetivo de ratio {target_positive_ratio:.3f} ya satisfecho "
                f"(actual {current_ratio:.3f}, tolerancia ±{tolerance:.3f})."
            )
        elif current_ratio > target_positive_ratio:
            if neg_idx.size == 0:
                print(
                    "⚠️  No se puede reducir la positividad al objetivo; no hay ventanas negativas disponibles."
                )
            else:
                drop_side = "positive"
                P_total = int(pos_idx.size)
                N_total = int(neg_idx.size)

                def _clamp_pos(value: int) -> int:
                    value = max(0, min(value, P_total))
                    if P_total > 0 and value == 0:
                        return 1
                    return value

                if target_low <= 0.0:
                    min_keep = 0
                else:
                    min_keep = math.ceil((target_low / (1.0 - target_low)) * N_total)
                if target_high >= 1.0:
                    max_keep = P_total
                else:
                    max_keep = math.floor((target_high / (1.0 - target_high)) * N_total)
                min_keep = _clamp_pos(min_keep)
                max_keep = _clamp_pos(max_keep)

                desired_center = _clamp_pos(int(round((target_positive_ratio / (1.0 - target_positive_ratio)) * N_total)))
                candidate_counts: set[int] = {
                    _clamp_pos(min_keep),
                    _clamp_pos(max_keep),
                    desired_center,
                    _clamp_pos(min_keep - 1),
                    _clamp_pos(max_keep + 1),
                    P_total,
                }

                best_keep = P_total
                best_ratio = current_ratio
                best_within = within_initial
                best_diff = abs(current_ratio - target_positive_ratio)

                for cand in sorted(candidate_counts):
                    total = cand + N_total
                    if total <= 0:
                        continue
                    ratio_cand = cand / total
                    within = target_low <= ratio_cand <= target_high
                    diff = abs(ratio_cand - target_positive_ratio)
                    better = False
                    if within and not best_within:
                        better = True
                    elif within == best_within:
                        if diff < best_diff - 1e-9:
                            better = True
                        elif math.isclose(diff, best_diff, rel_tol=1e-9, abs_tol=1e-9):
                            if cand > best_keep:
                                better = True
                    if better:
                        best_keep = cand
                        best_ratio = ratio_cand
                        best_within = within
                        best_diff = diff

                if best_keep < P_total:
                    if best_keep <= 0:
                        selected_pos = np.empty((0,), dtype=pos_idx.dtype)
                    else:
                        selected_pos = np.sort(rng.choice(pos_idx, size=best_keep, replace=False))
                    selected_indices = np.concatenate([selected_pos, neg_idx])
                    achieved_ratio = best_ratio
                else:
                    achieved_ratio = best_ratio
        else:
            if pos_idx.size == 0:
                print(
                    "⚠️  No se puede incrementar la positividad al objetivo; no hay ventanas positivas disponibles."
                )
            else:
                drop_side = "negative"
                P_total = int(pos_idx.size)
                N_total = int(neg_idx.size)

                def _clamp_neg(value: int) -> int:
                    value = max(0, min(value, N_total))
                    return value

                if target_high >= 1.0:
                    min_keep = 0
                else:
                    min_keep = math.ceil(P_total * (1.0 - target_high) / target_high)
                if target_low <= 0.0:
                    max_keep = N_total
                else:
                    max_keep = math.floor(P_total * (1.0 - target_low) / target_low)
                min_keep = _clamp_neg(min_keep)
                max_keep = _clamp_neg(max_keep)

                desired_center = _clamp_neg(int(round(P_total * (1.0 - target_positive_ratio) / target_positive_ratio)))
                candidate_counts: set[int] = {
                    _clamp_neg(min_keep),
                    _clamp_neg(max_keep),
                    desired_center,
                    _clamp_neg(min_keep - 1),
                    _clamp_neg(max_keep + 1),
                    N_total,
                }

                best_keep = N_total
                best_ratio = current_ratio
                best_within = within_initial
                best_diff = abs(current_ratio - target_positive_ratio)

                for cand in sorted(candidate_counts):
                    total = P_total + cand
                    if total <= 0:
                        continue
                    ratio_cand = P_total / total
                    within = target_low <= ratio_cand <= target_high
                    diff = abs(ratio_cand - target_positive_ratio)
                    better = False
                    if within and not best_within:
                        better = True
                    elif within == best_within:
                        if diff < best_diff - 1e-9:
                            better = True
                        elif math.isclose(diff, best_diff, rel_tol=1e-9, abs_tol=1e-9):
                            if cand > best_keep:
                                better = True
                    if better:
                        best_keep = cand
                        best_ratio = ratio_cand
                        best_within = within
                        best_diff = diff

                if best_keep < N_total:
                    if best_keep <= 0:
                        selected_neg = np.empty((0,), dtype=neg_idx.dtype)
                    else:
                        selected_neg = np.sort(rng.choice(neg_idx, size=best_keep, replace=False))
                    selected_indices = np.concatenate([pos_idx, selected_neg])
                    achieved_ratio = best_ratio
                else:
                    achieved_ratio = best_ratio

        if selected_indices is not None:
            selected_indices = np.sort(selected_indices)
            if selected_indices.size != total_before:
                sequences = sequences[selected_indices]
                labels = labels[selected_indices]
                patients = patients[selected_indices]
                records_arr = records_arr[selected_indices]
                if feature_array is not None:
                    feature_array = feature_array[selected_indices]
                window_flags = window_flags[selected_indices]
                undersampling_applied = True
                dropped_windows = total_before - int(selected_indices.size)
                pos_windows_final = int(window_flags.sum())
                neg_windows_final = int(window_flags.size - pos_windows_final)
                achieved_ratio = float(window_flags.mean()) if window_flags.size else 0.0
                undersampling_info = {
                    "target_ratio": target_positive_ratio,
                    "tolerance": tolerance,
                    "achieved_ratio": achieved_ratio,
                    "dropped_windows": dropped_windows,
                    "kept_windows": int(window_flags.size),
                    "dropped_class": drop_side,
                }
                print(
                    "   ↳ Undersampling objetivo: ventanas="
                    f"{window_flags.size} (positivas={pos_windows_final}, negativas={neg_windows_final}), "
                    f"ratio final={achieved_ratio:.3f} (previo {current_ratio:.3f})."
                )
            else:
                achieved_ratio = current_ratio
        elif not within_initial and drop_side is not None:
            print(
                f"⚠️  No se pudo ajustar la positividad al objetivo {target_positive_ratio:.3f}; "
                f"ratio actual {achieved_ratio:.3f} permanece fuera de la tolerancia ±{tolerance:.3f}."
            )

    seizure_windows = count_seizure_windows(labels, label_mode=label_mode)
    frame_matrix = _flatten_labels_to_frames(labels, label_mode=label_mode)
    total_frames = int(frame_matrix.size)
    positive_frames = int(frame_matrix.sum())
    pos_ratio = _positive_frame_ratio(labels, label_mode=label_mode)
    if pos_ratio < min_positive_ratio:
        print(
            f"⚠️  Positividad global {pos_ratio:.3f} inferior al umbral {min_positive_ratio:.3f}."
        )

    artifacts_meta: dict[str, object] = {}
    if storage_mode_final == "memmap":
        artifacts_meta.update(memmap_artifacts or {})
        artifacts_meta.setdefault("windows", int(sequences.shape[0]))
        artifacts_meta.setdefault("label_shape", labels.shape[1:] if labels.ndim > 1 else ())
        if promote_threshold_bytes is not None:
            artifacts_meta.setdefault("auto_memmap_threshold_bytes", int(promote_threshold_bytes))
        if parallel_workers:
            artifacts_meta.setdefault("feature_parallel_workers", int(parallel_workers))
    else:
        artifacts_meta = {}
    if sampling_info is not None:
        artifacts_meta["sampling"] = sampling_info
    if undersampling_info is not None:
        artifacts_meta["undersampling"] = undersampling_info

    if write_enabled and patient_acc is not None and cache_base_dir is not None:
        for patient_id in patients_needing_export:
            bundle = patient_acc.get(patient_id)
            if not bundle or not bundle["sequences"]:
                continue
            patient_path = _cache_file_path(cache_base_dir, "patient", cache_split_name, patient_id)
            seq_array = np.concatenate(bundle["sequences"], axis=0)
            label_array = np.concatenate(bundle["labels"], axis=0)
            if label_mode == "window":
                label_array = label_array.astype(np.int32, copy=False)
            else:
                label_array = label_array.astype(np.float32, copy=False)
            patient_ids = [patient_id] * int(label_array.shape[0])
            record_ids = list(bundle["records"])
            features_array = None
            if include_features and bundle["features"]:
                features_array = np.concatenate(bundle["features"], axis=0)
            bundle_hop = bundle.get("hop_samples")
            if bundle_hop is None:
                bundle_hop = expected_hop_samples
            bundle_target_fs = bundle.get("target_fs")
            if bundle_target_fs is None:
                bundle_target_fs = target_fs
            _write_np_cache(
                patient_path,
                sequences=seq_array,
                labels=label_array,
                patients=patient_ids,
                records=record_ids,
                label_mode=label_mode,
                hop_samples=int(bundle_hop) if bundle_hop is not None else None,
                target_fs=float(bundle_target_fs) if bundle_target_fs is not None else None,
                features=features_array,
                feature_names=feature_keys if include_features else None,
                compression=compression if use_npz_compression else None,
            )
            print(f"   ↳ Caché de paciente guardada: {patient_path}")

    unique_patients_count = np.unique(patients).size
    if label_mode == "time_step":
        print(
            f"\n📦 Dataset final -> ventanas: {labels.shape[0]}, ventanas positivas: {seizure_windows}, "
            f"frames: {total_frames} (positivos {positive_frames}), pacientes únicos: {unique_patients_count}, "
            f"positividad frames: {pos_ratio:.3f}"
        )
    else:
        print(
            f"\n📦 Dataset final -> ventanas: {labels.shape[0]}, convulsiones: {seizure_windows}, "
            f"pacientes únicos: {unique_patients_count}, positividad global: {pos_ratio:.3f}"
        )

    return DatasetBundle(
        sequences=sequences,
        features=feature_array,
        feature_names=feature_names,
        labels=labels,
        patients=patients,
        records=records_arr,
        label_mode=label_mode,
        storage=storage_mode_final,
        artifacts=artifacts_meta,
    )


def collect_records_for_split(config: PipelineConfig, split: str) -> tuple[list[Path], dict[str, int]]:
    split_key = split.lower()
    if split_key not in SPLIT_ATTRS:
        raise ValueError(f"Split desconocido: {split}")

    roots_attr, max_attr = SPLIT_ATTRS[split_key]
    roots: list[Path] = getattr(config, roots_attr) or []
    roots = [Path(p).expanduser().resolve() for p in roots]
    max_records: int = getattr(config, max_attr) or 0

    if max_records <= 0:
        print(f"⚠️  Se pidió 0 registros para el split '{split_key}'.")
        return [], {}

    if split_key == "train" and config.records:
        manual = [Path(p).expanduser().resolve() for p in config.records]
        if not manual:
            return [], {}
        records = manual[:max_records]
        print(
            f"Se utilizarán {len(records)} registros manuales para el split '{split_key}'"
        )
        return records, {}

    discovered, split_counts = discover_records_multi(
        roots,
        max_records=max_records,
        max_per_patient=config.max_per_patient,
        include_background_only=config.include_background_only_records,
        montage=config.montage,
    )
    pos_attr = f"max_records_{split_key}_positive"
    neg_attr = f"max_records_{split_key}_negative"
    positive_quota = getattr(config, pos_attr, None)
    negative_quota = getattr(config, neg_attr, None)

    records = consolidate_records_with_quota(
        discovered,
        [],
        max_records=max_records,
        max_per_patient=config.max_per_patient,
        positive_quota=positive_quota,
        negative_quota=negative_quota,
    )
    if not records:
        message = (
            f"No se encontraron registros válidos para el split '{split_key}'. Revisa las rutas o incrementa "
            f"'max_records_{split_key}'."
        )
        if split_key == "train":
            raise RuntimeError(message)
        print(f"⚠️  {message}")
        return [], split_counts

    unique_patients = len({p.stem.split("_")[0] for p in records})
    roots_display = ", ".join(str(r) for r in roots)
    print(
        f"Split '{split_key}': {len(records)} registros (pacientes únicos={unique_patients}) | raíces: {roots_display}"
    )
    if split_counts:
        breakdown = ", ".join(f"{split_name}:{count}" for split_name, count in sorted(split_counts.items()))
        print(f"   Distribución interna: {breakdown}")
    return records, split_counts
