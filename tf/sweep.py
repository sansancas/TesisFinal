#!/usr/bin/env python
"""Hyperparameter sweep driver for the TensorFlow EEG pipeline.

This orchestration layer launches sequential training trials with the
``tf.pipeline`` entry point while keeping GPU/CPU memory usage stable. It
creates per-trial configuration files derived from a base config, executes the
pipeline, extracts the metrics it writes, and maintains a consolidated
leaderboard.

Key features
============
- **Resource-friendly** execution: one trial at a time, explicit calls to
  ``tf.keras.backend.clear_session()`` and CUDA memory cleanup between runs,
  optional shared memmap cache for datasets, and automatic GPU memory-growth
  configuration when possible.
- **Flexible search spaces**: grid or random search with choice/uniform/
  log-uniform parameters mapped directly onto ``PipelineConfig`` attributes.
- **Persistent artefacts**: generated configs, pipeline run directories, CSV
  and JSON summaries, plus a plain-text log detailing progress and top trials.
- **Dry-run support**: inspect the generated parameter combinations without
  launching any training jobs.

Search space definition
=======================
Provide a JSON (or YAML, if ``pyyaml`` is installed) file describing the search
space::

    {
      "parameters": [
        {"name": "loss_type", "values": ["binary_crossentropy", "focal", "tversky"]},
        {"name": "focal_alpha", "values": [0.25, 0.35]},
        {"name": "learning_rate", "type": "loguniform", "min": 5e-5, "max": 5e-3},
        {"name": "lr_schedule_type", "values": ["plateau", "cosine"]},
        {"name": "cosine_annealing_min_lr", "type": "loguniform", "min": 1e-6, "max": 1e-4},
        {"name": "cosine_annealing_period", "values": [6, 10, 14]},
        {"name": "optimizer", "values": ["adam", "adamw"]},
        {"name": "optimizer_weight_decay", "values": [0.0, 1e-4, 5e-4]},
        {"name": "batch_size", "values": [8, 12, 16]},
        {"name": "dropout", "values": [0.2, 0.3, 0.4]}
      ],
      "mode": "random",
      "max_trials": 16,
      "metric": "eval_pr_auc"
    }

If ``search`` is omitted on the CLI, the script falls back to a compact default
space covering losses, optimisers, schedulers, and a few architecture knobs.

Command line usage::

    python -m tf.hparam_sweep \
        config=tf_config.json \
        search=search_space.json \
        output=runs/sweeps/tf_demo \
        metric=eval_pr_auc \
        max_trials=12

CLI arguments (key=value syntax)
--------------------------------
``config``        Base configuration file consumed by ``tf.pipeline`` (required).
``search``        JSON/YAML search space description (optional).
``output``        Sweep output directory (defaults to ``runs/sweeps/<timestamp>``).
``metric``        Metric key used for ranking trials (default: ``eval_pr_auc``).
``mode``          ``grid`` or ``random``.
``max_trials``    Number of trials (required for random mode; optional for grid).
``max_workers``   Concurrency level (currently executes sequentially; value >1 is ignored).
``seed``          Random seed for reproducible sampling (default: 42).
``top_k``         Number of best trials echoed in the completion log (default: 5).
``dry_run``       Print trial plan without executing the pipeline.
``keep_checkpoints``  Preserve per-trial checkpoints (default: False, only metrics kept).
``reuse_dataset_cache`` Reuse a shared memmap cache between trials (default: True).

Artefacts emitted under ``<output>``
------------------------------------
- ``configs/trial_*.json``: auto-generated trial configs.
- ``runs/``: directories produced by each pipeline execution.
- ``summary.csv`` / ``summary.json``: consolidated leaderboard with metrics.
- ``logs.txt``: append-only log with progress and top-k results.
"""

from __future__ import annotations

import copy
import csv
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:  # Optional YAML support
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

import tensorflow as tf

from utils import PipelineConfig, config_to_dict, load_config, validate_config


# ---------------------------------------------------------------------------
# Dataclasses and search-space modelling
# ---------------------------------------------------------------------------


@dataclass
class ParameterSpec:
    name: str
    kind: str  # "choice", "uniform", "loguniform"
    values: Optional[List[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    dtype: str = "auto"  # "auto", "int", "float", "bool"

    def requires_sampling(self) -> bool:
        return self.kind != "choice"

    def iter_grid(self) -> List[Any]:
        if self.kind != "choice" or self.values is None:
            raise ValueError(f"Parameter '{self.name}' is not discrete; cannot generate a grid.")
        return list(self.values)

    def sample(self, rng: random.Random) -> Any:
        if self.kind == "choice":
            if not self.values:
                raise ValueError(f"Parameter '{self.name}' has an empty choice list.")
            value = rng.choice(self.values)
        elif self.kind == "uniform":
            assert self.minimum is not None and self.maximum is not None
            value = rng.uniform(self.minimum, self.maximum)
        elif self.kind == "loguniform":
            assert self.minimum is not None and self.maximum is not None
            log_min = math.log(self.minimum)
            log_max = math.log(self.maximum)
            value = math.exp(rng.uniform(log_min, log_max))
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown parameter kind '{self.kind}'.")

        if self.dtype == "int":
            return int(round(value))
        if self.dtype == "float":
            return float(value)
        if self.dtype == "bool":
            return bool(value)
        if self.dtype == "auto" and isinstance(value, float):
            return float(value)
        return value


@dataclass
class SweepOptions:
    base_config_path: Path
    search_space_path: Optional[Path]
    output_dir: Path
    project_root: Path
    metric_key: str = "eval_pr_auc"
    mode: Optional[str] = None
    max_trials: Optional[int] = None
    max_workers: int = 1
    seed: int = 42
    top_k: int = 5
    dry_run: bool = False
    keep_checkpoints: bool = False
    reuse_dataset_cache: bool = True


@dataclass
class TrialResult:
    trial_id: int
    params: Dict[str, Any]
    status: str
    metric_value: Optional[float]
    metric_key: str
    run_dir: Optional[Path]
    summary_path: Optional[Path]
    log_path: Optional[Path] = None
    extra_metrics: Dict[str, float] = field(default_factory=dict)
    duration_sec: float = 0.0
    error_excerpt: Optional[str] = None


DEFAULT_SEARCH_SPACE: Dict[str, Any] = {
    "parameters": [
        {"name": "loss_type", "values": ["binary_crossentropy", "focal", "tversky"]},
        {"name": "focal_alpha", "values": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]},
        {"name": "focal_gamma", "values": [1.5, 2.0, 2.5]},
        {"name": "tversky_alpha", "values": [0.5, 0.6, 0.7, 0.8, 0.9]},
        {"name": "tversky_beta", "values": [0.1, 0.2, 0.3, 0.4, 0.5]},
        {"name": "learning_rate", "type": "loguniform", "min": 5e-5, "max": 5e-3},
        {"name": "min_lr", "type": "loguniform", "min": 1e-6, "max": 1e-4},
        {"name": "lr_schedule_type", "values": ["plateau", "cosine"]},
        {"name": "cosine_annealing_period", "values": [4, 8, 12]},
        {"name": "cosine_annealing_min_lr", "type": "loguniform", "min": 1e-6, "max": 1e-4},
        {"name": "optimizer", "values": ["adam", "adamw"]},
        {"name": "optimizer_weight_decay", "values": [0.0, 1e-2, 5e-2, 1e-3, 5e-3,1e-4, 5e-4]},
        {"name": "optimizer_use_ema", "values": [False, True]},
        {"name": "optimizer_ema_momentum", "values": [0.95, 0.99, 0.995, 0.999]},
        {"name": "use_class_weights", "values": [True, False]},
        {"name": "batch_size", "values": [8, 16, 32]},
        {"name": "num_filters", "values": [32, 48, 64, 96]},
        {"name": "rnn_units", "values": [32, 48, 64, 96]},
        {"name": "dropout", "values": [0.1, 0.2, 0.3, 0.4]},
    ],
    "mode": "random",
    "max_trials": 40,
    "metric": "eval_pr_auc",
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _configure_gpu_memory_growth() -> None:
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception:  # pragma: no cover - defensive
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            continue


def _parse_cli(argv: Sequence[str]) -> SweepOptions:
    if not argv:
        raise SystemExit(__doc__)

    options: Dict[str, str] = {}
    for raw in argv:
        token = raw.strip()
        if not token:
            continue
        if token in {"-h", "--help", "help"}:
            raise SystemExit(__doc__)
        if "=" not in token:
            raise SystemExit(f"Invalid argument '{token}'. Expected key=value syntax.")
        key, value = token.split("=", 1)
        options[key.strip().lower()] = value.strip()

    def _pop_path(name: str, required: bool = False) -> Optional[Path]:
        value = options.pop(name, None)
        if value is None:
            if required:
                raise SystemExit(f"Missing required argument '{name}'.")
            return None
        return Path(value).expanduser().resolve()

    base_config_path = _pop_path("config", required=True)
    search_space_path = _pop_path("search")
    output_dir = _pop_path("output")

    metric_key = options.pop("metric", "eval_pr_auc")
    mode = options.pop("mode", None)
    max_trials = options.pop("max_trials", None)
    max_workers = int(options.pop("max_workers", "1"))
    seed = int(options.pop("seed", "42"))
    top_k = int(options.pop("top_k", "5"))
    dry_run = options.pop("dry_run", "false").lower() in {"1", "true", "yes", "on"}
    keep_checkpoints = options.pop("keep_checkpoints", "false").lower() in {"1", "true", "yes", "on"}
    reuse_dataset_cache = options.pop("reuse_dataset_cache", "true").lower() in {"1", "true", "yes", "on"}

    if options:
        extra = ", ".join(sorted(options.keys()))
        raise SystemExit(f"Unknown argument(s): {extra}")

    if output_dir is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        base_anchor = base_config_path.parent if base_config_path is not None else Path.cwd()
        output_dir = base_anchor / "runs" / "sweeps" / timestamp

    project_root = _discover_project_root(base_config_path)

    if mode is not None and mode.lower() not in {"grid", "random"}:
        raise SystemExit("'mode' must be 'grid' or 'random'.")

    parsed_max_trials: Optional[int]
    if max_trials is not None:
        parsed_max_trials = int(max_trials)
        if parsed_max_trials <= 0:
            raise SystemExit("'max_trials' must be > 0 when provided.")
    else:
        parsed_max_trials = None

    return SweepOptions(
        base_config_path=base_config_path,
        search_space_path=search_space_path,
        output_dir=output_dir,
        project_root=project_root,
        metric_key=metric_key,
        mode=mode.lower() if mode else None,
        max_trials=parsed_max_trials,
        max_workers=max(1, max_workers),
        seed=seed,
        top_k=max(1, top_k),
        dry_run=dry_run,
        keep_checkpoints=keep_checkpoints,
        reuse_dataset_cache=reuse_dataset_cache,
    )


def _load_json_or_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML search spaces.")
        return yaml.safe_load(text)
    return json.loads(text)


def _discover_project_root(base_config_path: Path) -> Path:
    if base_config_path.is_dir():
        start = base_config_path
    else:
        start = base_config_path.parent
    candidate = start
    while True:
        tf_pkg = candidate / "tf"
        dataset_py = candidate / "dataset.py"
        if tf_pkg.is_dir() and (tf_pkg / "__init__.py").exists() and dataset_py.exists():
            return candidate
        if candidate.parent == candidate:
            return start
        candidate = candidate.parent


def _parse_parameter(raw: Dict[str, Any]) -> ParameterSpec:
    if "name" not in raw:
        raise ValueError("Each parameter definition requires a 'name'.")
    name = str(raw["name"])
    dtype = str(raw.get("dtype", "auto")).lower()

    if "values" in raw:
        values = raw["values"]
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
            raise ValueError(f"Parameter '{name}' has an invalid 'values' entry.")
        values_list = list(values)
        return ParameterSpec(name=name, kind="choice", values=values_list, dtype=dtype)

    param_type = str(raw.get("type", "choice")).lower()
    if param_type == "choice":
        raise ValueError(f"Parameter '{name}' missing 'values' for choice type.")

    minimum = raw.get("min")
    maximum = raw.get("max")
    if minimum is None or maximum is None:
        raise ValueError(f"Parameter '{name}' requires 'min' and 'max'.")
    return ParameterSpec(name=name, kind=param_type, minimum=float(minimum), maximum=float(maximum), dtype=dtype)


def _load_search_space(options: SweepOptions) -> Tuple[List[ParameterSpec], str, int]:
    if options.search_space_path is None:
        space = DEFAULT_SEARCH_SPACE
    else:
        space = _load_json_or_yaml(options.search_space_path)

    parameters_raw = space.get("parameters")
    if not parameters_raw:
        raise ValueError("Search space must include a non-empty 'parameters' list.")

    parameter_specs = [_parse_parameter(entry) for entry in parameters_raw]
    mode = options.mode or str(space.get("mode", "random")).lower()
    if mode not in {"grid", "random"}:
        raise ValueError("Search space 'mode' must be 'grid' or 'random'.")

    max_trials = options.max_trials
    if mode == "random":
        max_trials = max_trials or int(space.get("max_trials", 0))
        if not max_trials or max_trials <= 0:
            raise ValueError("Random search requires 'max_trials' > 0.")
    else:  # grid
        if max_trials is None:
            total = 1
            for param in parameter_specs:
                if param.requires_sampling():
                    raise ValueError(
                        "Grid search cannot include continuous parameters. "
                        f"Parameter '{param.name}' uses distribution '{param.kind}'."
                    )
                total *= len(param.iter_grid())
            max_trials = total
        else:
            for param in parameter_specs:
                if param.requires_sampling():
                    raise ValueError(
                        "Grid search with explicit 'max_trials' still requires discrete parameters only."
                    )

    metric_key = options.metric_key or space.get("metric") or "eval_pr_auc"
    return parameter_specs, mode, max_trials


def _generate_grid_trials(params: List[ParameterSpec]) -> Iterable[Dict[str, Any]]:
    grids = [param.iter_grid() for param in params]
    names = [param.name for param in params]
    for combo in product(*grids):
        yield {name: value for name, value in zip(names, combo)}


def _generate_random_trials(
    params: List[ParameterSpec],
    count: int,
    rng: random.Random,
) -> Iterable[Dict[str, Any]]:
    for _ in range(count):
        yield {param.name: param.sample(rng) for param in params}


# ---------------------------------------------------------------------------
# Metric aggregation helpers
# ---------------------------------------------------------------------------


def _floatify(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _collect_fold_metrics(path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not path.exists():
        return metrics
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    if not rows:
        return metrics
    numeric_keys = [key for key in rows[0] if key != "fold"]
    for key in numeric_keys:
        values = [_floatify(row.get(key)) for row in rows]
        clean_values = [v for v in values if v is not None]
        if not clean_values:
            continue
        metrics[f"cv_{key}"] = sum(clean_values) / len(clean_values)
    return metrics


def _collect_summary_metrics(run_dir: Path) -> Tuple[Dict[str, float], Optional[Path]]:
    metrics: Dict[str, float] = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = None
        if isinstance(summary, dict):
            for prefix, key in (
                ("eval", "evaluation_metrics"),
                ("val", "validation_metrics"),
            ):
                section = summary.get(key)
                if isinstance(section, dict):
                    for metric_name, metric_value in section.items():
                        value = _floatify(metric_value)
                        if value is not None:
                            metrics[f"{prefix}_{metric_name}"] = value
            history_last = summary.get("history_last")
            if isinstance(history_last, dict):
                for metric_name, metric_value in history_last.items():
                    value = _floatify(metric_value)
                    if value is not None:
                        metrics[f"history_{metric_name}"] = value
            baseline_eval = summary.get("baseline_eval_metrics")
            if isinstance(baseline_eval, dict):
                for metric_name, metric_value in baseline_eval.items():
                    value = _floatify(metric_value)
                    if value is not None:
                        metrics[f"baseline_eval_{metric_name}"] = value
    metrics.update(_collect_fold_metrics(run_dir / "fold_metrics.csv"))
    return metrics, summary_path if summary_path.exists() else None


def _metric_from_dict(metrics: Dict[str, float], key: str) -> Optional[float]:
    if key in metrics:
        return metrics[key]
    if "." in key:
        return metrics.get(key.replace(".", "_"))
    aliases = {
        "val_pr_auc": ["validation_pr_auc", "history_val_pr_auc"],
        "eval_pr_auc": ["evaluation_pr_auc"],
    }
    if key in aliases:
        for alias in aliases[key]:
            if alias in metrics:
                return metrics[alias]
    return None


# ---------------------------------------------------------------------------
# Core sweep execution
# ---------------------------------------------------------------------------


def _ensure_dirs(base: Path) -> Dict[str, Path]:
    cfg_dir = base / "configs"
    runs_dir = base / "runs"
    logs_path = base / "logs.txt"
    cache_dir = base / "dataset_cache"
    trial_logs_dir = base / "trial_logs"
    base.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    trial_logs_dir.mkdir(parents=True, exist_ok=True)
    return {
        "configs": cfg_dir,
        "runs": runs_dir,
        "logs": logs_path,
        "cache": cache_dir,
        "trial_logs": trial_logs_dir,
    }


def _write_log(path: Path, message: str) -> None:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    with path.open("a", encoding="utf-8") as fp:
        fp.write(line)
    print(line, end="")


def _extract_error_excerpt(log_path: Path, *, max_lines: int = 60, max_chars: int = 800) -> Optional[str]:
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    lines = text.splitlines()
    if not lines:
        return None
    tail = lines[-max_lines:]
    trigger_idx = 0
    markers = ("Traceback", "Error", "Exception", "ResourceExhausted", "OOM")
    for idx, line in enumerate(tail):
        if any(marker in line for marker in markers):
            trigger_idx = max(0, idx - 3)
            break
    excerpt = "\n".join(tail[trigger_idx:])
    if len(excerpt) > max_chars:
        excerpt = excerpt[-max_chars:]
    return excerpt.strip() if excerpt else None


def _prepare_trial_config(
    base_config: PipelineConfig,
    params: Dict[str, Any],
    trial_id: int,
    directories: Dict[str, Path],
    options: SweepOptions,
) -> Tuple[PipelineConfig, Path]:
    config = copy.deepcopy(base_config)
    for key, value in params.items():
        if not hasattr(config, key):
            raise AttributeError(f"PipelineConfig has no attribute '{key}'.")
        setattr(config, key, value)

    if "seed" not in params:
        config.seed = base_config.seed + trial_id
        
    if str(config.loss_type).lower() in {"focal", "tversky", "tversky_focal"} and config.use_class_weights:
        config.use_class_weights = False
        _write_log(
            directories["logs"],
            (
                f"   ↳ Trial {trial_id:03d}: desactivando 'use_class_weights' porque "
                f"loss_type={config.loss_type} requiere entrenamiento sin pesos."
            ),
        )

    runs_dir = directories["runs"]
    config.output_dir = runs_dir
    if not options.keep_checkpoints:
        config.save_metric_checkpoints = False
        config.checkpoint_dir = None
    else:
        checkpoints_root = runs_dir / "checkpoints"
        checkpoints_root.mkdir(parents=True, exist_ok=True)
        config.checkpoint_dir = checkpoints_root

    if options.reuse_dataset_cache:
        cache_dir = directories["cache"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        if config.dataset_storage in {"auto", "memmap"}:
            config.dataset_memmap_dir = cache_dir
    else:
        config.dataset_memmap_dir = None

    config.epoch_time_log_path = None

    trial_cfg_path = directories["configs"] / f"trial_{trial_id:03d}.json"
    trial_cfg_path.write_text(json.dumps(config_to_dict(config), indent=2, sort_keys=True), encoding="utf-8")
    return config, trial_cfg_path


def _discover_new_run_dir(base_runs_dir: Path, before: set[Path]) -> Optional[Path]:
    current = {path for path in base_runs_dir.iterdir() if path.is_dir()}
    new_dirs = [path for path in current if path not in before]
    if new_dirs:
        return max(new_dirs, key=lambda p: p.stat().st_mtime)
    if current:
        return max(current, key=lambda p: p.stat().st_mtime)
    return None


def _clear_tf_state() -> None:
    tf.keras.backend.clear_session()
    gc.collect()


def _run_single_trial(
    trial_id: int,
    params: Dict[str, Any],
    base_config: PipelineConfig,
    directories: Dict[str, Path],
    options: SweepOptions,
    metric_key: str,
) -> TrialResult:
    _clear_tf_state()
    config, cfg_path = _prepare_trial_config(base_config, params, trial_id, directories, options)
    config = validate_config(config)

    runs_dir = directories["runs"]
    before_runs = {path for path in runs_dir.iterdir() if path.is_dir()}

    log_path = directories["trial_logs"] / f"trial_{trial_id:03d}.log"
    candidate_scripts = [
        options.project_root / "tf" / "pipeline.py",
        options.project_root / "pipeline.py",
    ]
    if options.project_root.name == "tf":
        candidate_scripts.append(options.project_root.parent / "tf" / "pipeline.py")

    pipeline_script: Optional[Path] = None
    for candidate in candidate_scripts:
        if candidate is not None and candidate.exists():
            pipeline_script = candidate
            break

    if pipeline_script is None:
        raise FileNotFoundError(
            "No se encontró 'pipeline.py'. Revisá que exista en 'tf/pipeline.py' o en el directorio raíz del proyecto."
        )

    command = [
        sys.executable,
        str(pipeline_script),
        str(cfg_path),
    ]
    env = os.environ.copy()
    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts: List[str] = []
    for candidate in (
        str(options.project_root),
        str(pipeline_script.parent),
        str(pipeline_script.parent.parent),
    ):
        if candidate and os.path.isdir(candidate) and candidate not in pythonpath_parts:
            pythonpath_parts.append(candidate)
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    start_time = time.perf_counter()
    exit_code: Optional[int]
    try:
        with log_path.open("w", encoding="utf-8") as log_fp:
            log_fp.write("# Command: {}\n".format(" ".join(command)))
            log_fp.write("# Working dir: {}\n".format(options.project_root))
            log_fp.write("# Config path: {}\n".format(cfg_path))
            log_fp.write("# Pipeline script: {}\n\n".format(pipeline_script))
            log_fp.flush()
            completed = subprocess.run(
                command,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=options.project_root,
                check=False,
            )
        exit_code = completed.returncode
    except Exception as exc:  # pylint: disable=broad-except
        exit_code = -1
        with log_path.open("a", encoding="utf-8") as log_fp:
            log_fp.write(f"\n[SWEEP] Error launching trial: {exc}\n")
        _write_log(directories["logs"], f"Trial {trial_id:03d} crashed before start: {exc}")
    duration = time.perf_counter() - start_time

    run_dir = _discover_new_run_dir(runs_dir, before_runs)
    metrics: Dict[str, float]
    summary_path: Optional[Path]
    metrics, summary_path = ({}, None)
    metric_value: Optional[float] = None
    status = "ok" if exit_code == 0 else f"error({exit_code})"
    error_excerpt = _extract_error_excerpt(log_path) if status != "ok" else None

    if status == "ok" and run_dir is not None:
        metrics, summary_path = _collect_summary_metrics(run_dir)
        metric_value = _metric_from_dict(metrics, metric_key)
    elif status == "ok":
        status = "error(no_run_dir)"

    _clear_tf_state()

    return TrialResult(
        trial_id=trial_id,
        params=params,
        status=status,
        metric_value=metric_value,
        metric_key=metric_key,
        run_dir=run_dir,
        summary_path=summary_path,
        log_path=log_path,
        extra_metrics=metrics,
        duration_sec=duration,
        error_excerpt=error_excerpt,
    )


def _serialise_results(results: List[TrialResult], directories: Dict[str, Path]) -> None:
    if not results:
        return
    csv_path = directories["runs"].parent / "summary.csv"
    json_path = directories["runs"].parent / "summary.json"

    param_names = sorted({key for result in results for key in result.params.keys()})
    metric_names = sorted({key for result in results for key in result.extra_metrics.keys()})

    header = [
        "trial_id",
        "status",
        "metric_key",
        "metric_value",
        "duration_sec",
        "run_dir",
        "summary_path",
        "log_path",
        "error_excerpt",
    ] + [f"param_{name}" for name in param_names] + metric_names

    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        for result in results:
            row = [
                result.trial_id,
                result.status,
                result.metric_key,
                result.metric_value if result.metric_value is not None else "",
                round(result.duration_sec, 3),
                str(result.run_dir) if result.run_dir else "",
                str(result.summary_path) if result.summary_path else "",
                str(result.log_path) if result.log_path else "",
                (result.error_excerpt or "").replace("\n", " | "),
            ]
            row.extend(result.params.get(name, "") for name in param_names)
            row.extend(result.extra_metrics.get(name, "") for name in metric_names)
            writer.writerow(row)

    ordered = sorted(
        results,
        key=lambda r: (r.metric_value if r.metric_value is not None else float("-inf")),
        reverse=True,
    )
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "metric_key": results[0].metric_key,
        "results": [
            {
                "trial_id": result.trial_id,
                "status": result.status,
                "metric": result.metric_value,
                "run_dir": str(result.run_dir) if result.run_dir else None,
                "params": result.params,
                "metrics": result.extra_metrics,
                "duration_sec": result.duration_sec,
                "log_path": str(result.log_path) if result.log_path else None,
                "error_excerpt": result.error_excerpt,
            }
            for result in ordered
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _summarise_completion(results: List[TrialResult], directories: Dict[str, Path], options: SweepOptions) -> None:
    if not results:
        return
    successful = [r for r in results if r.metric_value is not None]
    if not successful:
        _write_log(directories["logs"], "No successful trials with metric values were produced.")
        return
    top_k = sorted(successful, key=lambda r: r.metric_value, reverse=True)[: options.top_k]
    summary_lines = [
        f"   • Trial {res.trial_id:03d}: metric={res.metric_value:.4f} | params={res.params}"
        for res in top_k
    ]
    _write_log(
        directories["logs"],
        "Top results (metric: {}):\n{}".format(results[0].metric_key, "\n".join(summary_lines)),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = _parse_cli(sys.argv[1:] if argv is None else argv)
    directories = _ensure_dirs(options.output_dir)

    rng = random.Random(options.seed)

    try:
        parameter_specs, mode, max_trials = _load_search_space(options)
    except Exception as exc:
        _write_log(directories["logs"], f"Error loading search space: {exc}")
        return 2

    try:
        base_config = validate_config(load_config(options.base_config_path))
    except Exception as exc:
        _write_log(directories["logs"], f"Error loading base config: {exc}")
        return 2

    _configure_gpu_memory_growth()

    _write_log(
        directories["logs"],
        (
            f"Sweep starting | mode={mode} | max_trials={max_trials} | metric={options.metric_key} | "
            f"output={options.output_dir}"
        ),
    )

    if options.dry_run:
        _write_log(directories["logs"], "Dry-run enabled; printing trial plans without execution.")

    if mode == "grid":
        trial_plan = list(_generate_grid_trials(parameter_specs))
        rng.shuffle(trial_plan)
    else:
        trial_plan = list(_generate_random_trials(parameter_specs, max_trials, rng))

    if options.dry_run:
        for idx, params in enumerate(trial_plan, start=1):
            _write_log(directories["logs"], f"Trial {idx:03d} params={params}")
        return 0

    results: List[TrialResult] = []
    for idx, params in enumerate(trial_plan, start=1):
        _write_log(directories["logs"], f"→ Trial {idx:03d}/{len(trial_plan)} | params={params}")
        result = _run_single_trial(
            trial_id=idx,
            params=params,
            base_config=base_config,
            directories=directories,
            options=options,
            metric_key=options.metric_key,
        )
        results.append(result)
        metric_display = (
            f"metric={result.metric_value:.4f}" if result.metric_value is not None else "metric=NA"
        )
        _write_log(
            directories["logs"],
            f"   Trial {idx:03d} finished | status={result.status} | {metric_display} | "
            f"duration={result.duration_sec/60.0:.2f} min",
        )
        if result.status != "ok":
            hint = "see log for details"
            if result.error_excerpt:
                first_line = result.error_excerpt.strip().splitlines()[0]
                hint = first_line if first_line else hint
            if result.log_path:
                _write_log(
                    directories["logs"],
                    f"      ↳ Failure hint: {hint}",
                )
                _write_log(
                    directories["logs"],
                    f"      ↳ Log file: {result.log_path}",
                )
            else:
                _write_log(
                    directories["logs"],
                    f"      ↳ Failure hint: {hint}",
                )
        _serialise_results(results, directories)

    _summarise_completion(results, directories, options)
    _write_log(directories["logs"], "Sweep completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
