import contextlib
from pathlib import Path
from dataclasses import asdict, dataclass, field, fields
from typing import Callable, Iterable, Iterator, Sequence
import json
import math
import pandas as pd

PREPROCESS = {
    'bandpass': (0.5, 40.),   # (low, high) Hz
    'notch': 60.,             # Hz (base or list)
    'notch_harmonics': True,
    'n_harmonics': 2,
    'resample': 256,          # Hz target
}

TIME_STEP = True
TRANSPOSE = True
NOTCH = True
BANDPASS = True
NORMALIZE = True

DEFAULT_WINDOW_SEC = 10.0
DEFAULT_HOP_SEC = 5.0
DEFAULT_EPS = 1e-8
DEFAULT_TARGET_FS = float(PREPROCESS.get("resample", 256.0) or 256.0)
DEFAULT_CONFIG_FILENAME = "nn_pipeline_config.json"

DEFAULT_DATA_ROOTS = [
    Path("/home/sansan/projects/TFDev/DATA_EEG_TUH/tuh_eeg_seizure/v2.0.3/edf/train"),
    Path("/home/sansan/projects/TFDev/DATA_EEG_TUH/tuh_eeg_seizure/v2.0.3/edf/dev"),
    Path("/home/sansan/projects/TFDev/DATA_EEG_TUH/tuh_eeg_seizure/v2.0.3/edf/eval"),
]

DEFAULT_SPLIT_ROOTS: dict[str, list[Path]] = {
    "train": [DEFAULT_DATA_ROOTS[0]],
    "val": [DEFAULT_DATA_ROOTS[1]],
    "eval": [DEFAULT_DATA_ROOTS[2]],
}

PREPROCESS_CONFIGS = {
    "filtered": {"bandpass": True, "notch": True, "normalize": True},
    "filters_only": {"bandpass": True, "notch": True, "normalize": False},
    "normalize_only": {"bandpass": False, "notch": False, "normalize": True},
    "unfiltered": {"bandpass": False, "notch": False, "normalize": False},
}

SPLIT_ATTRS = {
    "train": ("train_roots", "max_records_train"),
    "val": ("val_roots", "max_records_val"),
    "eval": ("eval_roots", "max_records_eval"),
}

@dataclass
class PipelineConfig:
    model: str = "hybrid"
    mode: str = "cv"
    data_roots: list[Path] | None = field(default_factory=list)
    train_roots: list[Path] | None = field(default_factory=list)
    val_roots: list[Path] | None = field(default_factory=list)
    eval_roots: list[Path] | None = field(default_factory=list)
    records: list[Path] | None = field(default_factory=list)
    max_records: int = 40
    max_records_train: int | None = None
    max_records_val: int | None = None
    max_records_eval: int | None = None
    max_per_patient: int = 1
    include_background_only_records: bool = True
    condition: str = "filtered"
    sampling_strategy: str = "none"
    sampling_seed: int | None = None
    montage: str = "ar"
    include_features: bool = False
    selected_features: list[str] = field(default_factory=list)
    time_step_labels: bool = False
    batch_size: int = 8
    epochs: int = 30
    folds: int = 5
    num_filters: int = 64
    kernel_size: int = 7
    dropout: float = 0.3
    rnn_units: int = 64
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    optimizer_weight_decay: float = 0.0
    optimizer_use_ema: bool = False
    optimizer_ema_momentum: float = 0.99
    jit_compile: bool = False
    window_sec: float = DEFAULT_WINDOW_SEC
    hop_sec: float = DEFAULT_HOP_SEC
    epsilon: float = DEFAULT_EPS
    target_fs: float = DEFAULT_TARGET_FS
    max_training_minutes: float | None = None
    preprocess_bandpass: bool | None = None
    preprocess_notch: bool | None = None
    preprocess_normalize: bool | None = None
    preprocess_n_harmonics: int | None = None
    use_class_weights: bool = True
    loss_type: str = "binary_crossentropy"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    tversky_alpha: float = 0.7
    tversky_beta: float = 0.3
    tversky_gamma: float = 1.3333333333
    patience: int = 5
    min_lr: float = 1e-5
    lr_schedule_type: str = "plateau"
    cosine_annealing_period: int = 10
    cosine_annealing_min_lr: float | None = None
    save_metric_checkpoints: bool = True
    seed: int = 42
    dry_run: bool = False
    output_dir: Path | None = None
    epoch_time_log_path: Path | None = None
    checkpoint_dir: Path | None = None
    verbose: int = 1
    final_validation_split: float = 0.0
    use_tf_dataset: bool = False
    tf_data_shuffle_buffer: int | None = None
    tf_data_prefetch: int | None = None
    tf_data_cache: str | bool | None = None
    write_tfrecords: bool = False
    tfrecord_dir: Path | None = None
    tfrecord_compression: str | None = None
    reuse_existing_tfrecords: bool = True
    dataset_storage: str = "ram"
    dataset_memmap_dir: Path | None = None


@dataclass
class TFRecordExportConfig:
    base_dir: Path
    split_name: str
    compression: str | None = None
    write_enabled: bool = False
    reuse_enabled: bool = False


def _as_path_list(values: Iterable[str] | None) -> list[Path]:
    if not values:
        return []
    return [Path(v).expanduser().resolve() for v in values]


def load_config(config_path: str | Path | None) -> PipelineConfig:
    candidate = Path(config_path) if config_path else Path(DEFAULT_CONFIG_FILENAME)
    candidate = candidate.expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {candidate}")

    with candidate.open("r", encoding="utf-8") as fp:
        raw_cfg = json.load(fp)

    config_kwargs: dict[str, object] = {}
    for fdef in fields(PipelineConfig):
        if fdef.name not in raw_cfg:
            continue
        value = raw_cfg[fdef.name]
        if fdef.name in {"data_roots", "records", "train_roots", "val_roots", "eval_roots"}:
            value = _as_path_list(value)
        elif fdef.name == "output_dir" and value:
            value = Path(value).expanduser().resolve()
        elif fdef.name == "tfrecord_dir" and value:
            value = Path(value).expanduser().resolve()
        elif fdef.name == "epoch_time_log_path" and value:
            value = Path(value).expanduser().resolve()
        elif fdef.name == "checkpoint_dir" and value:
            value = Path(value).expanduser().resolve()
        elif fdef.name == "dataset_memmap_dir" and value:
            value = Path(value).expanduser().resolve()
        elif fdef.name == "preprocess_n_harmonics" and value is not None:
            value = int(value)
        elif fdef.name == "selected_features" and value is not None:
            value = [str(item) for item in value]
        elif fdef.name == "dataset_storage" and value is not None:
            value = str(value).lower().strip()
        elif fdef.name == "max_training_minutes" and value is not None:
            value = float(value)
        config_kwargs[fdef.name] = value

    config = PipelineConfig(**config_kwargs)

    config.data_roots = [Path(p).expanduser().resolve() for p in (config.data_roots or [])]
    if not config.data_roots:
        config.data_roots = [p.resolve() for p in DEFAULT_DATA_ROOTS]

    config.train_roots = [Path(p).expanduser().resolve() for p in (config.train_roots or [])]
    config.val_roots = [Path(p).expanduser().resolve() for p in (config.val_roots or [])]
    config.eval_roots = [Path(p).expanduser().resolve() for p in (config.eval_roots or [])]

    if not config.train_roots:
        config.train_roots = [
            p
            for p in config.data_roots
            if any(keyword in {part.lower() for part in p.parts} for keyword in ("train",))
        ]
    if not config.train_roots:
        config.train_roots = [p.resolve() for p in DEFAULT_SPLIT_ROOTS["train"]]

    if not config.val_roots:
        config.val_roots = [
            p
            for p in config.data_roots
            if {part.lower() for part in p.parts} & {"val", "valid", "validation", "dev"}
        ]
    if not config.val_roots:
        config.val_roots = [p.resolve() for p in DEFAULT_SPLIT_ROOTS["val"]]

    if not config.eval_roots:
        config.eval_roots = [
            p for p in config.data_roots if {part.lower() for part in p.parts} & {"eval", "evaluation", "test"}
        ]
    if not config.eval_roots:
        config.eval_roots = [p.resolve() for p in DEFAULT_SPLIT_ROOTS["eval"]]

    if config.max_records_train is None:
        config.max_records_train = config.max_records
    if config.max_records_val is None:
        config.max_records_val = config.max_records
    if config.max_records_eval is None:
        config.max_records_eval = config.max_records

    if config.records:
        config.records = [Path(p).expanduser().resolve() for p in config.records]
    else:
        config.records = []

    config.selected_features = [str(item) for item in (config.selected_features or [])]

    return config


def config_to_dict(config: PipelineConfig) -> dict[str, object]:
    raw = asdict(config)
    for key, value in list(raw.items()):
        if isinstance(value, Path):
            raw[key] = str(value)
        elif isinstance(value, list):
            raw[key] = [str(item) if isinstance(item, Path) else item for item in value]
    return raw


def validate_config(config: PipelineConfig) -> PipelineConfig:
    if config.model not in {"tcn", "hybrid"}:
        raise ValueError(f"Modelo inválido '{config.model}'. Usa 'tcn' o 'hybrid'.")
    if config.mode not in {"cv", "final"}:
        raise ValueError("'mode' debe ser 'cv' o 'final'.")
    if config.condition not in PREPROCESS_CONFIGS:
        raise ValueError(
            f"Condición '{config.condition}' inválida. Opciones: {sorted(PREPROCESS_CONFIGS)}"
        )
    if config.mode == "cv" and config.folds < 2:
        raise ValueError("Se requieren al menos 2 folds para la validación cruzada.")
    if config.batch_size <= 0:
        raise ValueError("'batch_size' debe ser > 0")
    if config.epochs <= 0:
        raise ValueError("'epochs' debe ser > 0")
    if config.learning_rate <= 0:
        raise ValueError("'learning_rate' debe ser > 0")
    if config.max_training_minutes is not None:
        if config.max_training_minutes <= 0:
            raise ValueError("'max_training_minutes' debe ser > 0 cuando se especifica.")
    optimizer_name = config.optimizer.lower()
    if optimizer_name not in {"adam", "adamw"}:
        raise ValueError("'optimizer' debe ser 'adam' o 'adamw'.")
    config.optimizer = optimizer_name
    if config.optimizer_weight_decay < 0:
        raise ValueError("'optimizer_weight_decay' debe ser >= 0.")
    if not (0.0 <= config.optimizer_ema_momentum < 1.0):
        raise ValueError("'optimizer_ema_momentum' debe estar en [0.0, 1.0).")
    if not isinstance(config.jit_compile, bool):
        raise ValueError("'jit_compile' debe ser True o False.")
    if config.window_sec <= 0:
        raise ValueError("'window_sec' debe ser > 0.")
    if config.hop_sec <= 0:
        raise ValueError("'hop_sec' debe ser > 0.")
    if config.epsilon <= 0:
        raise ValueError("'epsilon' debe ser > 0.")
    if config.target_fs <= 0:
        raise ValueError("'target_fs' debe ser > 0.")
    for attr_name in ("preprocess_bandpass", "preprocess_notch", "preprocess_normalize"):
        attr_value = getattr(config, attr_name)
        if attr_value is not None and not isinstance(attr_value, bool):
            raise ValueError(f"'{attr_name}' debe ser True, False o None.")
    if config.preprocess_n_harmonics is not None and config.preprocess_n_harmonics < 0:
        raise ValueError("'preprocess_n_harmonics' debe ser >= 0 cuando se especifica.")
    if not isinstance(config.use_class_weights, bool):
        raise ValueError("'use_class_weights' debe ser True o False.")
    schedule_type = config.lr_schedule_type.lower()
    if schedule_type not in {"plateau", "cosine"}:
        raise ValueError("'lr_schedule_type' debe ser 'plateau' o 'cosine'.")
    config.lr_schedule_type = schedule_type
    if config.cosine_annealing_period <= 0:
        raise ValueError("'cosine_annealing_period' debe ser > 0.")
    if config.cosine_annealing_min_lr is not None and config.cosine_annealing_min_lr <= 0:
        raise ValueError("'cosine_annealing_min_lr' debe ser > 0 cuando se especifica.")
    if not isinstance(config.save_metric_checkpoints, bool):
        raise ValueError("'save_metric_checkpoints' debe ser True o False.")
    loss_name = config.loss_type.lower()
    allowed_losses = {"binary_crossentropy", "focal", "tversky", "tversky_focal"}
    if loss_name not in allowed_losses:
        raise ValueError(f"'loss_type' debe estar en {sorted(allowed_losses)}.")
    config.loss_type = loss_name
    if not (0.0 <= config.focal_alpha <= 1.0):
        raise ValueError("'focal_alpha' debe estar en [0, 1].")
    if config.focal_gamma <= 0:
        raise ValueError("'focal_gamma' debe ser > 0.")
    if not (0.0 <= config.tversky_alpha <= 1.0):
        raise ValueError("'tversky_alpha' debe estar en [0, 1].")
    if not (0.0 <= config.tversky_beta <= 1.0):
        raise ValueError("'tversky_beta' debe estar en [0, 1].")
    if config.tversky_gamma <= 0:
        raise ValueError("'tversky_gamma' debe ser > 0.")
    if config.max_records <= 0:
        raise ValueError("'max_records' debe ser > 0")
    if config.max_per_patient <= 0:
        raise ValueError("'max_per_patient' debe ser > 0")
    if not isinstance(config.include_background_only_records, bool):
        raise ValueError("'include_background_only_records' debe ser True o False.")
    if not isinstance(config.selected_features, list):
        raise ValueError("'selected_features' debe ser una lista (posiblemente vacía) de strings.")
    if any(not isinstance(name, str) for name in config.selected_features):
        raise ValueError("Todos los elementos de 'selected_features' deben ser strings.")
    if config.selected_features and not config.include_features:
        raise ValueError("'selected_features' requiere 'include_features': true.")
    if not isinstance(config.time_step_labels, bool):
        raise ValueError("'time_step_labels' debe ser True o False.")
    sampling_strategy = str(config.sampling_strategy).lower().strip()
    allowed_sampling = {"none", "balanced"}
    if sampling_strategy not in allowed_sampling:
        raise ValueError("'sampling_strategy' debe ser uno de: 'none', 'balanced'.")
    config.sampling_strategy = sampling_strategy
    if config.sampling_seed is not None:
        try:
            config.sampling_seed = int(config.sampling_seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("'sampling_seed' debe ser un entero o None.") from exc
    if not (0.0 <= config.final_validation_split < 1.0):
        raise ValueError("'final_validation_split' debe estar en [0.0, 1.0).")
    if not config.train_roots:
        raise ValueError("Debes especificar al menos un directorio en 'train_roots'.")
    if config.max_records_train is None or config.max_records_train <= 0:
        raise ValueError("'max_records_train' debe ser > 0.")
    for split_name in ("val", "eval"):
        roots_attr, max_attr = SPLIT_ATTRS[split_name]
        roots = getattr(config, roots_attr)
        max_records_split = getattr(config, max_attr)
        if roots is None:
            raise ValueError(f"Debes especificar una lista (posiblemente vacía) en '{roots_attr}'.")
        if max_records_split is None or max_records_split < 0:
            raise ValueError(f"'{max_attr}' debe ser >= 0.")
    if config.tf_data_shuffle_buffer is not None and config.tf_data_shuffle_buffer <= 0:
        raise ValueError("'tf_data_shuffle_buffer' debe ser > 0 o None.")
    if config.tf_data_prefetch is not None and config.tf_data_prefetch < 0:
        raise ValueError("'tf_data_prefetch' debe ser >= 0 o None.")
    if config.tf_data_cache is not None and not isinstance(config.tf_data_cache, (str, bool)):
        raise ValueError("'tf_data_cache' debe ser bool, string o None.")
    if config.write_tfrecords and config.tfrecord_dir is None:
        raise ValueError("Debes especificar 'tfrecord_dir' cuando 'write_tfrecords' es true.")
    if config.tfrecord_compression and config.tfrecord_compression.upper() not in {"GZIP", "ZLIB"}:
        raise ValueError("'tfrecord_compression' debe ser 'GZIP', 'ZLIB' o None.")
    if config.tfrecord_compression:
        config.tfrecord_compression = config.tfrecord_compression.upper()
    if not isinstance(config.reuse_existing_tfrecords, bool):
        raise ValueError("'reuse_existing_tfrecords' debe ser True o False.")
    storage_mode = str(config.dataset_storage).lower().strip()
    if storage_mode not in {"ram", "memmap"}:
        raise ValueError("'dataset_storage' debe ser 'ram' o 'memmap'.")
    config.dataset_storage = storage_mode
    if storage_mode == "memmap" and config.sampling_strategy != "none":
        raise ValueError("'dataset_storage'='memmap' requiere 'sampling_strategy': 'none'.")
    if config.dataset_memmap_dir is not None and not isinstance(config.dataset_memmap_dir, Path):
        raise ValueError("'dataset_memmap_dir' debe ser una ruta válida o None.")
    if config.time_step_labels:
        if config.write_tfrecords:
            raise ValueError("'time_step_labels' no es compatible con 'write_tfrecords'.")
        if config.reuse_existing_tfrecords:
            raise ValueError("'time_step_labels' no es compatible con 'reuse_existing_tfrecords'.")
    return config

# -----------------------------------------------------------------------------
# Utility helpers (borrowed from the notebook logic)
# -----------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    average_precision: float
    roc_auc: float

@contextlib.contextmanager
def apply_preprocess_config(config: dict, *, target_fs: float) -> Iterator[None]:
    """Temporarily override dataset2 preprocessing flags."""
    global BANDPASS, NOTCH, NORMALIZE
    original_flags = (BANDPASS, NOTCH, NORMALIZE)
    original_harmonics = PREPROCESS.get("n_harmonics", 0)
    bandpass = bool(config.get("bandpass", False))
    notch = bool(config.get("notch", False))
    normalize = bool(config.get("normalize", False))
    requested_harmonics = config.get("n_harmonics")
    try:
        BANDPASS = bandpass
        NOTCH = notch
        NORMALIZE = normalize
        if notch:
            base = float(PREPROCESS.get("notch", 0.0) or 0.0)
            if base > 0.0:
                nyq = target_fs / 2.0
                max_h = max(0, int(math.floor(nyq / base) - 1))
                if requested_harmonics is None:
                    PREPROCESS["n_harmonics"] = min(int(original_harmonics), max_h)
                else:
                    PREPROCESS["n_harmonics"] = min(int(requested_harmonics), max_h)
            else:
                PREPROCESS["n_harmonics"] = int(requested_harmonics or 0)
        else:
            PREPROCESS["n_harmonics"] = int(requested_harmonics or 0)
        yield
    finally:
        BANDPASS, NOTCH, NORMALIZE = original_flags
        PREPROCESS["n_harmonics"] = original_harmonics

def save_cv_outputs(
    *,
    output_dir: Path,
    config: PipelineConfig,
    fold_results: list[FoldResult],
    predictions_df: pd.DataFrame,
    feature_names: list[str],
    run_id: str | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame([asdict(fr) for fr in fold_results])
    metrics_path = output_dir / "fold_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    summary_path = output_dir / "summary.json"
    summary = {
        "config": config_to_dict(config),
        "fold_metrics_mean": metrics_df.drop(columns=["fold"]).mean(numeric_only=True).to_dict(),
        "fold_metrics_std": metrics_df.drop(columns=["fold"]).std(numeric_only=True).fillna(0.0).to_dict(),
        "feature_names": feature_names,
    }
    if run_id is not None:
        summary["run_id"] = run_id
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    preds_path = output_dir / "predictions.csv"
    predictions_df.to_csv(preds_path, index=False)
    print(f"\n📝 Resultados CV guardados en {output_dir}")


def resolve_preprocess_settings(config: PipelineConfig) -> dict[str, object]:
    base = PREPROCESS_CONFIGS.get(config.condition, {}).copy()
    base.setdefault("bandpass", False)
    base.setdefault("notch", False)
    base.setdefault("normalize", False)
    default_harmonics = int(PREPROCESS.get("n_harmonics", 0) or 0)
    base["n_harmonics"] = default_harmonics

    if config.preprocess_bandpass is not None:
        base["bandpass"] = bool(config.preprocess_bandpass)
    if config.preprocess_notch is not None:
        base["notch"] = bool(config.preprocess_notch)
    if config.preprocess_normalize is not None:
        base["normalize"] = bool(config.preprocess_normalize)
    if config.preprocess_n_harmonics is not None:
        base["n_harmonics"] = int(config.preprocess_n_harmonics)

    return base


def resolve_checkpoint_dir(config: PipelineConfig, run_id: str) -> Path:
    if config.checkpoint_dir is not None:
        base = config.checkpoint_dir
    elif config.output_dir is not None:
        base = config.output_dir / "checkpoints"
    else:
        base = Path.cwd() / "runs" / "checkpoints" / run_id
    return base


def resolve_epoch_time_log_path(config: PipelineConfig, run_id: str) -> Path:
    base_dir: Path | None = config.output_dir
    custom = config.epoch_time_log_path
    if custom is not None:
        custom_path = Path(custom)
        if custom_path.is_absolute():
            path = custom_path
        else:
            target_base = base_dir if base_dir is not None else (Path.cwd() / "runs" / run_id)
            path = target_base / custom_path
    elif base_dir is not None:
        path = base_dir / "epoch_times.csv"
    else:
        path = Path.cwd() / "runs" / run_id / "epoch_times.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path