import os
import gc
import glob
import tempfile
from uuid import uuid4
import numpy as np
import tensorflow as tf
import mne
import pandas as pd
import math
from pathlib import Path
from collections import defaultdict
from typing import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, fields
import hashlib
from utils import TFRecordExportConfig, apply_preprocess_config, PipelineConfig, SPLIT_ATTRS, PREPROCESS, BANDPASS, NOTCH, NORMALIZE

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# =================================================================================================================
# Montage-based CSV listing & filtering
# =================================================================================================================

def detect_suffix_strict(ch_names):
    """Detect channel name suffix across all channels robustly.
    Returns '-LE' if majority end with '-LE', '-REF' if majority '-REF',
    otherwise tries to infer by presence of canonical names.
   """
    le = sum(1 for ch in ch_names if ch.endswith('-LE'))
    rf = sum(1 for ch in ch_names if ch.endswith('-REF'))
    if le > rf and le > 0:
        return '-LE'
    if rf > le and rf > 0:
        return '-REF'
    # fallback: look for canonical bipolar references
    for suf in ('-LE','-REF'):
        for base in ('F7','F8','T3','T4','T5','T6','C3','C4','P3','P4','O1','O2'):
            if any(ch == base + suf for ch in ch_names):
                return suf
    # default to -REF if unknown
    return '-REF'

# Definir montajes como constantes para evitar recrear en cada llamada
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
    ]
}


def _notch_freqs_from_cfg(cfg):
    notch = cfg.get('notch', None)
    if notch in (None, 0, False):
        return None
    if isinstance(notch, (list, tuple)):
        return [float(f) for f in notch if f]
    base = float(notch)
    n_h = int(cfg.get('n_harmonics', 0)) if cfg.get('notch_harmonics', False) else 0
    return [base * (i+1) for i in range(n_h + 1)]


def preprocess_edf(raw, config=PREPROCESS):
    """
    Optimized preprocessing function with better memory management and early returns.
    """
    # Early return if no processing needed
    if not any([BANDPASS, NOTCH, NORMALIZE]):
        if config.get('resample', 0) != raw.info['sfreq']:
            raw.resample(config['resample'], npad='auto', verbose=False)
        return raw
    
    # Apply filters before resampling to avoid aliasing
    if BANDPASS:
        raw.filter(config['bandpass'][0], config['bandpass'][1], method='fir', phase='zero', verbose=False)
    
    if NOTCH:
        freqs = _notch_freqs_from_cfg(config)
        if freqs:
            raw.notch_filter(
                freqs=freqs,
                method='fir',
                phase='zero',
                verbose=False
            )

    # Resample after filtering
    if config.get('resample', 0) and config['resample'] != raw.info['sfreq']:
        raw.resample(config['resample'], npad='auto', verbose=False)
    
    # Normalize last to work with final sampling rate
    if NORMALIZE:
        # More efficient normalization using vectorized operations
        raw.apply_function(
            lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-6),
            picks='eeg',
            channel_wise=False  # Process all channels at once for better vectorization
        )
    
    return raw

def extract_montage_signals(edf_path: str, montage: str='ar', desired_fs: int=0):
    """
    Optimized function to read EDF and return montage-specific bipolar signals.
    Returns raw_bip object with bipolar montage applied.
    """
    # Validate montage early
    if montage not in MONTAGE_PAIRS:
        raise ValueError(f"Unsupported montage '{montage}'. Use 'ar' or 'le'.")
    
    # Read EDF with minimal memory footprint initially
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    
    # Determine suffix efficiently
    ch_names_set = set(raw.ch_names)
    suf = detect_suffix_strict(raw.ch_names)
    
    # Build pairs with suffix
    pairs = [(f'EEG {a}{suf}', f'EEG {b}{suf}') for a, b in MONTAGE_PAIRS[montage]]
    
    # Check for missing channels early (before loading data)
    needed = {c for pair in pairs for c in pair}
    missing = needed - ch_names_set
    if missing:
        raw.close()  # Clean up
        raise RuntimeError(f"Missing required electrodes for {montage} montage: {missing}")
    
    # Only load data after validation
    raw.load_data()
    
    # Apply preprocessing
    raw = preprocess_edf(raw, config=PREPROCESS)
    
    # Create bipolar montage more efficiently
    anodes = [a for a, _ in pairs]
    cathodes = [b for _, b in pairs]
    ch_names_bip = [f"{a}-{b}" for a, b in pairs]
    
    raw_bip = mne.set_bipolar_reference(
        raw,
        anode=anodes,
        cathode=cathodes,
        ch_name=ch_names_bip,
        drop_refs=True,
        verbose=False
    )
    
    # Pick only the bipolar channels we created
    raw_bip.pick(ch_names_bip)
    
    # Final resampling if needed
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


def discover_records(
    base_dir: Path,
    *,
    max_records: int,
    max_per_patient: int,
    include_background_only: bool = False,
) -> list[Path]:
    if not base_dir.exists():
        return []
    csv_files = sorted(base_dir.rglob("*_bi.csv"))
    ordered_candidates: list[tuple[Path, str, bool]] = []
    seen: set[str] = set()
    positive_candidates = 0
    background_candidates_total = 0
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
        edf_path = next((p for p in candidates if p.exists()), None)
        if edf_path is None:
            continue
        key = str(edf_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        patient_id = edf_path.stem.split("_")[0]
        if has_seizure:
            positive_candidates += 1
        else:
            background_candidates_total += 1
        ordered_candidates.append((edf_path, patient_id, has_seizure))
        if not include_background_only and positive_candidates >= max_records:
            break

    selected: list[Path] = []
    patient_counts: dict[str, int] = defaultdict(int)
    background_selected = 0

    for path, patient_id, has_seizure in ordered_candidates:
        if patient_counts[patient_id] >= max_per_patient:
            continue
        selected.append(path)
        patient_counts[patient_id] += 1
        if not has_seizure:
            background_selected += 1
        if len(selected) >= max_records:
            break

    if include_background_only and background_candidates_total:
        if background_selected == 0:
            print(
                f"ℹ️  Se identificaron {background_candidates_total} registros solo fondo en '{base_dir}',"
                f" pero no se añadieron por llegar al límite max_records={max_records}."
            )
        else:
            print(
                f"ℹ️  Registros solo fondo incluidos desde '{base_dir}': {background_selected}"
                f" (de {background_candidates_total} candidatos)."
            )

    return selected


def discover_records_multi(
    base_dirs: Iterable[Path],
    *,
    max_records: int,
    max_per_patient: int,
    include_background_only: bool = False,
) -> tuple[list[Path], dict[str, int]]:
    collected: list[Path] = []
    seen: set[str] = set()
    patient_counts: dict[str, int] = defaultdict(int)
    split_counts: dict[str, int] = defaultdict(int)
    for base_dir in base_dirs:
        if len(collected) >= max_records:
            break
        remaining = max_records - len(collected)
        for path in discover_records(
            base_dir,
            max_records=remaining,
            max_per_patient=max_per_patient,
            include_background_only=include_background_only,
        ):
            key = str(path.resolve())
            if key in seen:
                continue
            patient_id = path.stem.split("_")[0]
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
            split_counts[split_name] += 1
            if len(collected) >= max_records:
                break
    return collected, dict(split_counts)


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
        if patient_counts[patient_id] >= max_per_patient:
            continue
        selected.append(path)
        seen.add(key)
        patient_counts[patient_id] += 1
        if len(selected) >= max_records:
            break
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
        # Flush current views
        self._sequence_mm.flush()
        self._label_mm.flush()
        if self._feature_mm is not None:
            self._feature_mm.flush()
        # Resize underlying files to exact size
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
        # Reopen with exact shapes (read-write to allow training updates like caching)
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
        # Update internal references
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

def _bytes_feature(values: Sequence[str | bytes]) -> tf.train.Feature:
    as_bytes = [v if isinstance(v, bytes) else v.encode("utf-8") for v in values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=as_bytes))


def _float_feature(values: np.ndarray | Sequence[float]) -> tf.train.Feature:
    if isinstance(values, np.ndarray):
        values = values.ravel().astype(np.float32)
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))


def _int64_feature(values: Sequence[int]) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v) for v in values]))


def load_record_from_tfrecord(
    path: Path,
    *,
    compression: str | None,
    include_features: bool,
    expected_length: int | None = None,
    expected_channels: int | None = None,
    expected_hop_samples: int | None = None,
    expected_target_fs: float | None = None,
) -> dict[str, object] | None:
    if not path.exists():
        return None

    if compression:
        dataset = tf.data.TFRecordDataset(str(path), compression_type=compression)
    else:
        dataset = tf.data.TFRecordDataset(str(path))

    feature_description: dict[str, tf.io.VarLenFeature | tf.io.FixedLenFeature] = {
        "sequence": tf.io.VarLenFeature(tf.float32),
        "sequence_length": tf.io.FixedLenFeature([], tf.int64),
        "sequence_channels": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "patient_id": tf.io.FixedLenFeature([], tf.string),
        "record_id": tf.io.FixedLenFeature([], tf.string),
        "features": tf.io.VarLenFeature(tf.float32),
        "features_length": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "hop_samples": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "target_fs": tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
    }

    sequences: list[np.ndarray] = []
    labels: list[int] = []
    patient_ids: list[str] = []
    record_ids: list[str] = []
    features_list: list[np.ndarray] = []

    has_examples = False
    metadata_checked = False
    stored_hop_samples: int | None = None
    stored_target_fs: float | None = None

    for raw_example in dataset:
        has_examples = True
        parsed = tf.io.parse_single_example(raw_example, feature_description)

        seq_len = int(parsed["sequence_length"].numpy())
        seq_channels = int(parsed["sequence_channels"].numpy())
        seq_dense = tf.sparse.to_dense(parsed["sequence"]).numpy()
        if seq_len * seq_channels != seq_dense.size:
            print(f"⚠️  TFRecord corrupta en {path}: tamaños inconsistentes.")
            return None
        if expected_length is not None and seq_len != expected_length:
            print(
                f"⚠️  TFRecord {path} contiene ventanas de longitud {seq_len}"
                f" diferente a la esperada ({expected_length}); se re-generará desde EDF."
            )
            return None
        if expected_channels is not None and seq_channels != expected_channels:
            print(
                f"⚠️  TFRecord {path} contiene {seq_channels} canales"
                f" pero se esperaban {expected_channels}; se re-generará desde EDF."
            )
            return None

        seq = seq_dense.reshape(seq_len, seq_channels)
        sequences.append(seq)

        hop_samples_val = int(parsed["hop_samples"].numpy())
        if hop_samples_val < 0:
            hop_samples_val = None
        target_fs_val = float(parsed["target_fs"].numpy())
        if target_fs_val < 0:
            target_fs_val = None

        if not metadata_checked:
            if expected_hop_samples is not None:
                if hop_samples_val is None:
                    print(
                        f"⚠️  TFRecord {path} carece de metadata de hop; se re-generará desde EDF."
                    )
                    return None
                if abs(hop_samples_val - expected_hop_samples) > 0:
                    print(
                        f"⚠️  TFRecord {path} usa hop_samples={hop_samples_val}"
                        f" distinto al esperado ({expected_hop_samples}); se re-generará desde EDF."
                    )
                    return None
            if expected_target_fs is not None:
                if target_fs_val is None:
                    print(
                        f"⚠️  TFRecord {path} carece de metadata de frecuencia; se re-generará desde EDF."
                    )
                    return None
                if not math.isclose(target_fs_val, expected_target_fs, rel_tol=1e-6, abs_tol=1e-6):
                    print(
                        f"⚠️  TFRecord {path} usa fs={target_fs_val:g}"
                        f" distinto al esperado ({expected_target_fs:g}); se re-generará desde EDF."
                    )
                    return None
            metadata_checked = True

        if stored_hop_samples is None and hop_samples_val is not None:
            stored_hop_samples = hop_samples_val
        if stored_target_fs is None and target_fs_val is not None:
            stored_target_fs = target_fs_val

        labels.append(int(parsed["label"].numpy()))
        patient_ids.append(parsed["patient_id"].numpy().decode("utf-8"))
        record_ids.append(parsed["record_id"].numpy().decode("utf-8"))

        if include_features:
            feat_len = int(parsed["features_length"].numpy())
            feat_sparse = parsed["features"]
            feat_values_count = int(tf.size(feat_sparse.values).numpy())
            feat_dense = tf.sparse.to_dense(feat_sparse).numpy() if feat_values_count else np.array([], dtype=np.float32)
            if feat_len == 0 or feat_dense.size == 0:
                print(f"⚠️  TFRecord {path} carece de features requeridas; se re-generará desde EDF.")
                return None
            features_list.append(feat_dense.reshape(feat_len))
        elif int(tf.size(parsed["features"].values).numpy()) > 0:
            # archivo contiene features pero no se requieren; se ignoran
            pass

    if not has_examples:
        print(f"⚠️  TFRecord vacía detectada en {path}; se ignorará.")
        return None

    result: dict[str, object] = {
        "sequences": sequences,
        "labels": np.asarray(labels, dtype=np.int32),
        "patients": patient_ids,
        "records": record_ids,
        "features": features_list if include_features else None,
        "hop_samples": stored_hop_samples,
        "target_fs": stored_target_fs,
    }
    return result


def build_tf_dataset_from_arrays(
    inputs: np.ndarray | tuple[np.ndarray, np.ndarray],
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    shuffle_buffer: int | None,
    prefetch: int | None,
    cache: str | bool | None,
):
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    if cache:
        if isinstance(cache, str):
            cache_path = Path(cache)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            dataset = dataset.cache(str(cache_path))
        else:
            dataset = dataset.cache()
    if shuffle:
        buffer = shuffle_buffer if shuffle_buffer is not None else labels.shape[0]
        buffer = max(1, min(int(buffer), labels.shape[0]))
        dataset = dataset.shuffle(
            buffer_size=buffer,
            seed=seed,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.batch(batch_size)
    if prefetch is None:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    elif prefetch > 0:
        dataset = dataset.prefetch(prefetch)
    return dataset

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


def prepare_model_inputs(
    sequence_array: np.ndarray,
    feature_array: np.ndarray | None,
    *,
    for_tf_dataset: bool = False,
):
    if feature_array is None:
        return sequence_array
    if for_tf_dataset:
        return (sequence_array, feature_array)
    return [sequence_array, feature_array]

def serialize_window_example(
    sequence: np.ndarray,
    label: int,
    *,
    patient_id: str,
    record_id: str,
    features: np.ndarray | None = None,
    hop_samples: int,
    target_fs: float,
) -> bytes:
    feature_map = {
        "sequence": _float_feature(sequence.astype(np.float32)),
        "sequence_length": _int64_feature([sequence.shape[0]]),
        "sequence_channels": _int64_feature([sequence.shape[1]]),
        "label": _int64_feature([int(label)]),
        "patient_id": _bytes_feature([patient_id]),
        "record_id": _bytes_feature([record_id]),
    }
    feature_map["hop_samples"] = _int64_feature([int(hop_samples)])
    feature_map["target_fs"] = _float_feature([float(target_fs)])
    if features is not None:
        feature_map["features"] = _float_feature(features.astype(np.float32))
        feature_map["features_length"] = _int64_feature([features.shape[-1]])
    example = tf.train.Example(features=tf.train.Features(feature=feature_map))
    return example.SerializeToString()

def write_tfrecord_file(
    path: Path,
    sequences: np.ndarray,
    labels: np.ndarray,
    patient_ids: Sequence[str],
    record_ids: Sequence[str],
    *,
    features: np.ndarray | None = None,
    compression: str | None = None,
    hop_samples: int,
    target_fs: float,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    options = None
    if compression:
        options = tf.io.TFRecordOptions(compression_type=compression.upper())
    with tf.io.TFRecordWriter(str(path), options=options) as writer:
        for idx in range(labels.shape[0]):
            seq = sequences[idx]
            label = int(labels[idx])
            feat_vec = features[idx] if features is not None else None
            example = serialize_window_example(
                seq,
                label,
                patient_id=patient_ids[idx],
                record_id=record_ids[idx],
                features=feat_vec,
                hop_samples=hop_samples,
                target_fs=target_fs,
            )
            writer.write(example)

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
    tfrecord_export: TFRecordExportConfig | None = None,
    storage_mode: str = "ram",
    memmap_dir: Path | str | None = None,
    memmap_prefix: str | None = None,
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
    use_memmap = storage_mode_normalized == "memmap"
    if storage_mode_normalized not in {"ram", "memmap"}:
        raise ValueError("'storage_mode' debe ser 'ram' o 'memmap'.")

    sampling_strategy_lower = (sampling_strategy or "none").lower()
    if use_memmap and sampling_strategy_lower != "none":
        raise ValueError(
            "Las estrategias de muestreo distintas de 'none' no están disponibles con storage_mode='memmap'."
        )

    if use_memmap:
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
    write_enabled = bool(tfrecord_export and tfrecord_export.write_enabled)
    reuse_enabled = bool(tfrecord_export and tfrecord_export.reuse_enabled)
    compression = tfrecord_export.compression if tfrecord_export else None
    base_dir = tfrecord_export.base_dir if tfrecord_export else None
    split_name = tfrecord_export.split_name if tfrecord_export else ""

    if time_step_labels and (write_enabled or reuse_enabled):
        raise RuntimeError(
            "'time_step_labels' no es compatible con la exportación o reutilización de TFRecords."
        )

    patient_acc: defaultdict[str, dict[str, list]] | None = None
    patients_needing_export: set[str] = set()

    memmap_builder: _MemmapAccumulator | None = None
    memmap_artifacts: dict[str, object] | None = None

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

    for edf_path in unique_records:
        csv_path = edf_path.with_name(edf_path.stem + "_bi.csv")
        intervals = read_intervals(csv_path)
        record_id = edf_path.stem
        patient_id = record_id.split("_")[0]

        record_path: Path | None = None
        patient_path: Path | None = None
        record_exists = False
        patient_exists = False
        if base_dir is not None:
            record_path = base_dir / "by_record" / split_name / f"{record_id}.tfrecord"
            patient_path = base_dir / "by_patient" / split_name / f"{patient_id}.tfrecord"
            record_exists = record_path.exists()
            patient_exists = patient_path.exists()
            if write_enabled and not patient_exists:
                patients_needing_export.add(patient_id)

        if not intervals:
            if not include_background_only:
                print(f"⚠️  {record_id}: sin intervalos de convulsión; omitido.")
                continue
            print(f"ℹ️  {record_id}: solo etiquetas de fondo; se incluirá con ventanas negativas.")

        if reuse_enabled and record_exists and record_path is not None:
            loaded = load_record_from_tfrecord(
                record_path,
                compression=compression,
                include_features=include_features,
                expected_length=expected_length,
                expected_hop_samples=expected_hop_samples,
                expected_target_fs=target_fs,
            )
            if loaded is not None:
                loaded_sequences = loaded["sequences"]
                loaded_labels = loaded["labels"]
                loaded_patients = loaded["patients"]
                loaded_records = loaded["records"]
                loaded_features = loaded.get("features") if include_features else None
                loaded_hop_samples = loaded.get("hop_samples")
                loaded_target_fs = loaded.get("target_fs")

                can_reuse = True
                if include_features:
                    if loaded_features is None:
                        can_reuse = False
                    else:
                        loaded_features = np.asarray(loaded_features, dtype=np.float32)
                        if requested_features is not None:
                            if feature_keys is None:
                                feature_keys = list(requested_features)
                            expected_dim = len(feature_keys)
                            if loaded_features.shape[1] != expected_dim:
                                print(
                                    f"⚠️  {record_id}: TFRecord contiene {loaded_features.shape[1]} features "
                                    f"pero se solicitaron {expected_dim}; se regenerará desde EDF."
                                )
                                can_reuse = False
                        elif feature_keys is None:
                            feature_keys = [f"feature_{idx}" for idx in range(loaded_features.shape[1])]

                if can_reuse:
                    reused_sequences = np.asarray(loaded_sequences, dtype=np.float32)
                    if time_step_labels:
                        reused_labels = np.asarray(loaded_labels, dtype=np.float32)
                        label_shape = reused_labels.shape[1:]
                        label_dtype = np.float32
                    else:
                        reused_labels = np.asarray(loaded_labels, dtype=np.int32)
                        label_shape = ()
                        label_dtype = np.int32

                    if include_features and loaded_features is not None:
                        reused_features = np.asarray(loaded_features, dtype=np.float32)
                        feature_dim = int(reused_features.shape[1])
                    else:
                        reused_features = None
                        feature_dim = None

                    if use_memmap:
                        if memmap_builder is None:
                            if memmap_root is None or memmap_prefix_value is None:
                                raise RuntimeError("Configuración inválida de memmap (ruta o prefijo no definidos).")

                            memmap_builder = _MemmapAccumulator(
                                base_dir=memmap_root,
                                prefix=memmap_prefix_value,
                                window_shape=reused_sequences.shape[1:],
                                label_shape=label_shape,
                                label_dtype=label_dtype,
                                include_features=include_features,
                                feature_dim=feature_dim,
                            )
                            memmap_artifacts = dict(memmap_builder.artifacts)
                        memmap_builder.append(reused_sequences, reused_labels, reused_features)
                    else:
                        sequence_batches.append(reused_sequences)
                        label_batches.append(reused_labels)
                        if include_features and reused_features is not None:
                            feature_batches.append(reused_features)

                    patient_list.extend(list(loaded_patients))
                    record_list.extend(list(loaded_records))

                    if write_enabled and patient_path is not None and not patient_exists and patient_acc is not None:
                        bundle = patient_acc[patient_id]
                        bundle["sequences"].append(reused_sequences)
                        bundle["labels"].append(reused_labels)
                        bundle["records"].extend(list(loaded_records))
                        if include_features and reused_features is not None:
                            bundle["features"].append(reused_features)
                        if bundle["hop_samples"] is None:
                            bundle["hop_samples"] = int(loaded_hop_samples) if loaded_hop_samples is not None else expected_hop_samples
                        elif loaded_hop_samples is not None and bundle["hop_samples"] != int(loaded_hop_samples):
                            print(
                                f"⚠️  {record_id}: hop_samples inconsistente entre registros del mismo paciente; "
                                "se usará el valor inicial."
                            )
                        if bundle["target_fs"] is None:
                            bundle["target_fs"] = float(loaded_target_fs) if loaded_target_fs is not None else float(target_fs)
                        elif loaded_target_fs is not None and not math.isclose(bundle["target_fs"], float(loaded_target_fs), rel_tol=1e-6, abs_tol=1e-6):
                            print(
                                f"⚠️  {record_id}: frecuencia objetivo inconsistente entre registros del mismo paciente; "
                                "se usará el valor inicial ({bundle['target_fs']:.6g})."
                            )

                    print(
                        f"✓ {record_id}: {loaded_labels.shape[0]} ventanas cargadas desde TFRecord "
                        f"(positividad {float(np.mean(loaded_labels)) if loaded_labels.size else 0.0:.3f})"
                    )
                    continue
            else:
                record_exists = False
                if write_enabled and patient_acc is not None:
                    patients_needing_export.add(patient_id)
                patient_exists = False

        raw = None
        try:
            with apply_preprocess_config(config, target_fs=target_fs):
                raw = extract_montage_signals(
                    str(edf_path), montage=montage, desired_fs=target_fs
                )
            data = raw.get_data()  # (channels, time)
            fs = float(raw.info["sfreq"])
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
        record_windows: list[np.ndarray] = []
        record_labels: list[int] = []
        feature_rows: list[np.ndarray] = []

        for start in range(0, data.shape[1] - window_samples + 1, hop_samples):
            end = start + window_samples
            window = data[:, start:end].T.astype(np.float32)  # (time, channels)
            segment_mask = mask[start:end]
            record_windows.append(window)
            if time_step_labels:
                record_labels.append(segment_mask.astype(np.float32))
            else:
                label = 1 if segment_mask.mean() >= 0.5 else 0
                record_labels.append(label)

            if include_features:
                feats = compute_feature_vector(window, fs, eps=eps)
                if requested_features is not None:
                    if feature_keys is None:
                        missing = [name for name in requested_features if name not in feats]
                        if missing:
                            raise ValueError(
                                "Las siguientes features solicitadas no están disponibles: "
                                + ", ".join(sorted(missing))
                            )
                        feature_keys = list(requested_features)
                    feature_rows.append(np.array([feats[k] for k in feature_keys], dtype=np.float32))
                else:
                    if feature_keys is None:
                        feature_keys = sorted(feats.keys())
                    feature_rows.append(np.array([feats[k] for k in feature_keys], dtype=np.float32))

        if not record_windows:
            print(f"⚠️  {record_id}: sin ventanas válidas después del preprocesamiento.")
            continue

        record_sequences = np.stack(record_windows, axis=0)
        if time_step_labels:
            record_labels_arr = np.stack(
                [np.asarray(vec, dtype=np.float32) for vec in record_labels], axis=0
            )
            label_dtype = np.float32
            label_shape = record_labels_arr.shape[1:]
        else:
            record_labels_arr = np.asarray(record_labels, dtype=np.int32)
            label_dtype = np.int32
            label_shape = ()

        if include_features:
            if not feature_rows:
                raise RuntimeError(
                    "Se solicitó include_features pero no se pudieron calcular features para al menos un registro."
                )
            record_features_arr = np.stack(feature_rows, axis=0)
            feature_dim = int(record_features_arr.shape[1])
        else:
            record_features_arr = None
            feature_dim = None

        if use_memmap:
            if memmap_builder is None:
                if memmap_root is None or memmap_prefix_value is None:
                    raise RuntimeError("Configuración inválida de memmap (ruta o prefijo no definidos).")
                memmap_builder = _MemmapAccumulator(
                    base_dir=memmap_root,
                    prefix=memmap_prefix_value,
                    window_shape=record_sequences.shape[1:],
                    label_shape=label_shape,
                    label_dtype=label_dtype,
                    include_features=include_features,
                    feature_dim=feature_dim,
                )
                memmap_artifacts = dict(memmap_builder.artifacts)
            memmap_builder.append(record_sequences, record_labels_arr, record_features_arr)
        else:
            sequence_batches.append(record_sequences)
            label_batches.append(record_labels_arr)
            if include_features and record_features_arr is not None:
                feature_batches.append(record_features_arr)

        patient_list.extend([patient_id] * record_sequences.shape[0])
        record_list.extend([record_id] * record_sequences.shape[0])

        if write_enabled and record_path is not None:
            record_patients = [patient_id] * record_sequences.shape[0]
            record_ids_per_window = [record_id] * record_sequences.shape[0]
            record_labels_export = record_labels_arr.astype(np.int32, copy=False)
            record_features_export = record_features_arr if include_features else None

            if not record_exists:
                write_tfrecord_file(
                    record_path,
                    record_sequences,
                    record_labels_export,
                    record_patients,
                    record_ids_per_window,
                    features=record_features_export,
                    compression=compression,
                    hop_samples=hop_samples,
                    target_fs=fs,
                )
                print(f"   ↳ TFRecord (registro) guardado: {record_path}")
            else:
                print(f"   ↳ TFRecord (registro) ya existe, se reutiliza: {record_path}")

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
                    f"⚠️  {record_id}: hop_samples inconsistente entre registros del mismo paciente; "
                    "se usará el valor inicial."
                )
            if patient_bundle["target_fs"] is None:
                patient_bundle["target_fs"] = float(fs)
            elif not math.isclose(patient_bundle["target_fs"], float(fs), rel_tol=1e-6, abs_tol=1e-6):
                print(
                    f"⚠️  {record_id}: frecuencia objetivo inconsistente entre registros del mismo paciente; "
                    f"se usará el valor inicial ({patient_bundle['target_fs']:.6g})."
                )

        print(
            f"✓ {record_id}: {len(record_labels)} ventanas generadas (positividad {np.mean(record_labels):.3f})"
        )

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
    sampling_strategy_lower = (sampling_strategy or "none").lower()
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
                print(
                    f"   ↳ Stratified sampling ('balanced'): ventanas={selected_idx.size}, "
                    f"positivos={pos_windows}, negativos={int(selected_idx.size - pos_windows)}"
                )
        else:
            print(
                f"⚠️  Estrategia de muestreo '{sampling_strategy_lower}' no soportada; se omite."
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
    else:
        artifacts_meta = {}

    if write_enabled and patient_acc is not None and base_dir is not None:
        patient_dir = base_dir / "by_patient" / split_name
        for patient_id in patients_needing_export:
            bundle = patient_acc.get(patient_id)
            if not bundle or not bundle["sequences"]:
                continue
            patient_path = patient_dir / f"{patient_id}.tfrecord"
            if patient_path.exists():
                print(f"   ↳ TFRecord (paciente) sobrescrito: {patient_path}")
            seq_array = np.concatenate(bundle["sequences"], axis=0)
            label_cat = np.concatenate(bundle["labels"], axis=0)
            label_array = label_cat.astype(np.int32, copy=False)
            record_ids = bundle["records"]
            patient_ids = [patient_id] * label_array.shape[0]
            features_array = None
            if include_features and bundle["features"]:
                features_array = np.concatenate(bundle["features"], axis=0)
            bundle_hop = bundle.get("hop_samples")
            if bundle_hop is None:
                if expected_hop_samples is None:
                    raise RuntimeError(
                        "No se encontró metadata de hop_samples para exportar TFRecord por paciente."
                    )
                bundle_hop = expected_hop_samples
            bundle_target_fs = bundle.get("target_fs")
            if bundle_target_fs is None:
                bundle_target_fs = target_fs
            if bundle_target_fs is None:
                raise RuntimeError(
                    "No se encontró metadata de frecuencia objetivo para exportar TFRecord por paciente."
                )
            write_tfrecord_file(
                patient_path,
                seq_array,
                label_array,
                patient_ids,
                record_ids,
                features=features_array,
                compression=compression,
                hop_samples=int(bundle_hop),
                target_fs=float(bundle_target_fs),
            )
            print(f"   ↳ TFRecord (paciente) guardado: {patient_path}")

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
    )
    records = consolidate_records(
        discovered,
        [],
        max_records=max_records,
        max_per_patient=config.max_per_patient,
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