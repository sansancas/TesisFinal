#!/usr/bin/env python
"""Hiperparámetro sweep orchestration for the PyTorch EEG pipeline.

This utility automates sequential hyperparameter exploration while keeping GPU
and CPU usage in check. It reuses the existing ``pt.pipeline`` entry point,
materialises a temporary configuration per trial, launches the run, gathers the
metrics written by the pipeline, and keeps a scoreboard with the best
configurations.

Highlights
==========
- **Resource-aware**: runs trials sequentially (default) and releases CUDA
  memory between trials, preventing resource saturation.
- **Flexible search**: supports grid and random search with choice/uniform/
  log-uniform parameters covering all ``PipelineConfig`` options (optimisers,
  learning rate schedules, losses, etc.).
- **Persistent results**: stores per-trial metadata and metrics in
  ``summary.json`` / ``summary.csv`` within the sweep directory, plus a
  human-readable log on stdout.
- **Dataset reuse**: by default, points all trials to a shared memmap cache to
  avoid rebuilding TFRecords/NPZ repeatedly.

Search-space specification
==========================
Provide a JSON file with the following structure (YAML works if ``pyyaml`` is
installed)::

    {
      "parameters": [
        {"name": "loss_type", "values": ["binary_crossentropy", "focal", "tversky"]},
        {"name": "focal_alpha", "values": [0.25, 0.35, 0.4]},
        {"name": "learning_rate", "type": "loguniform", "min": 1e-4, "max": 5e-3},
        {"name": "lr_schedule_type", "values": ["plateau", "cosine"]},
        {"name": "cosine_annealing_min_lr", "type": "loguniform", "min": 1e-6, "max": 1e-4},
        {"name": "cosine_annealing_period", "values": [6, 10, 14]},
        {"name": "min_lr", "type": "loguniform", "min": 1e-6, "max": 1e-4},
        {"name": "optimizer", "values": ["adam", "adamw"]},
        {"name": "optimizer_weight_decay", "values": [0.0, 1e-4, 5e-4]},
        {"name": "use_class_weights", "values": [true, false]},
        {"name": "dropout", "values": [0.2, 0.3, 0.4]}
      ],
      "mode": "random",              // or "grid"
      "max_trials": 16,               // optional for grid; required for random
      "metric": "eval_pr_auc"         // any metric collected by the pipeline
    }

When ``--search`` is omitted, a compact default search space focusing on loss
and scheduler hyperparameters is used.

Command-line usage (key=value syntax)::

    python -m hparam_sweep \
        config=pt_config.json \
        search=search_space.json \
        output=runs/sweeps/20251006 \
        metric=eval_pr_auc \
        max_trials=12

Key CLI arguments
-----------------
``config``        Path to the base configuration JSON (same format as
                  ``pt.pipeline`` uses).
``search``        Path to the search-space JSON/YAML. Optional.
``output``        Root directory for sweep artefacts. Defaults to
                  ``runs/sweeps/<timestamp>``.
``metric``        Metric key used to rank trials (default: ``eval_pr_auc``).
``mode``          ``grid`` or ``random``. Overrides search file.
``max_trials``    Maximum number of trials. Required for random mode; optional
                  for grid (falls back to the Cartesian product size).
``max_workers``   Concurrency level (defaults to 1; the script enforces
                  sequential execution for reliability).
``seed``          Random seed for reproducibility (default: 42).
``top_k``         Number of best trials to keep in the JSON summary (default: 5).
``dry_run``       If true, only prints the generated trial list.

The script writes the following artefacts under ``<output>``:

- ``configs/trial_*.json``: generated config files per trial.
- ``runs/``: the actual training runs produced by ``pt.pipeline``.
- ``summary.csv`` / ``summary.json``: consolidated leaderboard.
- ``logs.txt``: plain-text progress log.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import random
import sys
import time
import copy
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

import torch

import pipeline as pt_pipeline
from utils import PipelineConfig, config_to_dict, load_config, validate_config


# ---------------------------------------------------------------------------
# Dataclasses and helpers
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
    extra_metrics: Dict[str, float] = field(default_factory=dict)
    duration_sec: float = 0.0


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
        {"name": "dropout", "values": [0.1, 0.2, 0.3, 0.4]},
    ],
    "mode": "random",
    "max_trials": 40,
    "metric": "eval_pr_auc",
}


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


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
    search_space_path = _pop_path("search", required=False)
    output_dir = _pop_path("output", required=False)

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
        output_dir = Path.cwd() / "runs" / "sweeps" / timestamp

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


# ---------------------------------------------------------------------------
# Search-space handling
# ---------------------------------------------------------------------------


def _load_json_or_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML search spaces.")
        return yaml.safe_load(text)
    return json.loads(text)


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


def _load_search_space(options: SweepOptions) -> Tuple[List[ParameterSpec], str, Optional[int]]:
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
    else:  # grid mode
        if max_trials is None:
            # compute Cartesian product size
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
            # still ensure no continuous parameters
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
        mean_value = sum(clean_values) / len(clean_values)
        metrics[f"cv_{key}"] = mean_value
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
    # Fold metrics (CV mode)
    fold_metrics = _collect_fold_metrics(run_dir / "fold_metrics.csv")
    metrics.update(fold_metrics)
    return metrics, summary_path if summary_path.exists() else None


def _metric_from_dict(metrics: Dict[str, float], key: str) -> Optional[float]:
    if key in metrics:
        return metrics[key]
    # Allow dotted access (e.g., "eval.pr_auc")
    if "." in key:
        alt_key = key.replace(".", "_")
        return metrics.get(alt_key)
    # Try a few fallbacks
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
    for directory in (cfg_dir, runs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    base.mkdir(parents=True, exist_ok=True)
    return {
        "configs": cfg_dir,
        "runs": runs_dir,
        "logs": logs_path,
        "cache": cache_dir,
    }


def _write_log(path: Path, message: str) -> None:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    with path.open("a", encoding="utf-8") as fp:
        fp.write(line)
    print(line, end="")


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

    # Vary seed unless explicitly overridden
    if "seed" not in params:
        config.seed = base_config.seed + trial_id

    # Route outputs into the sweep directory
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
        config.dataset_storage = "memmap"
        config.dataset_memmap_dir = cache_dir
    else:
        config.dataset_memmap_dir = None

    config.epoch_time_log_path = None  # pipeline will place it under the run dir

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


def _run_single_trial(
    trial_id: int,
    params: Dict[str, Any],
    base_config: PipelineConfig,
    directories: Dict[str, Path],
    options: SweepOptions,
    metric_key: str,
) -> TrialResult:
    config, cfg_path = _prepare_trial_config(base_config, params, trial_id, directories, options)
    config = validate_config(config)  # ensure we fail fast before launching heavy work

    runs_dir = directories["runs"]
    before_runs = {path for path in runs_dir.iterdir() if path.is_dir()}

    start_time = time.perf_counter()
    exit_code = None
    try:
        exit_code = pt_pipeline.main([str(cfg_path)])
    except SystemExit as exc:  # pipeline shouldn't exit, but guard anyway
        exit_code = exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:  # pylint: disable=broad-except
        exit_code = -1
        _write_log(directories["logs"], f"Trial {trial_id:03d} crashed: {exc}")
    duration = time.perf_counter() - start_time

    run_dir = _discover_new_run_dir(runs_dir, before_runs)
    metrics, summary_path = ({}, None)
    metric_value: Optional[float] = None
    status = "ok" if exit_code == 0 else f"error({exit_code})"

    if status == "ok" and run_dir is not None:
        metrics, summary_path = _collect_summary_metrics(run_dir)
        metric_value = _metric_from_dict(metrics, metric_key)
    elif status == "ok":
        status = "error(no_run_dir)"

    torch.cuda.empty_cache()
    gc.collect()

    return TrialResult(
        trial_id=trial_id,
        params=params,
        status=status,
        metric_value=metric_value,
        metric_key=metric_key,
        run_dir=run_dir,
        summary_path=summary_path,
        extra_metrics=metrics,
        duration_sec=duration,
    )


def _serialise_results(results: List[TrialResult], directories: Dict[str, Path]) -> None:
    if not results:
        return
    csv_path = directories["runs"].parent / "summary.csv"
    json_path = directories["runs"].parent / "summary.json"

    # Determine union of parameter names and metrics
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
            ]
            row.extend(result.params.get(name, "") for name in param_names)
            row.extend(result.extra_metrics.get(name, "") for name in metric_names)
            writer.writerow(row)

    best_sorted = sorted(
        results,
        key=lambda r: (r.metric_value if r.metric_value is not None else float("-inf")),
        reverse=True,
    )
    summary_payload = {
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
            }
            for result in best_sorted
        ],
    }
    json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")


def _summarise_completion(results: List[TrialResult], directories: Dict[str, Path], options: SweepOptions) -> None:
    if not results:
        return
    best_sorted = sorted(
        [r for r in results if r.metric_value is not None],
        key=lambda r: r.metric_value,  # type: ignore[arg-type]
        reverse=True,
    )
    if not best_sorted:
        _write_log(directories["logs"], "No successful trials with metric values were produced.")
        return
    top_k = best_sorted[: options.top_k]
    _write_log(
        directories["logs"],
        "Top results (metric: {}):\n{}".format(
            results[0].metric_key,
            "\n".join(
                f"   • Trial {res.trial_id:03d}: metric={res.metric_value:.4f} | params={res.params}"
                for res in top_k
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = _parse_cli(sys.argv[1:] if argv is None else argv)
    directories = _ensure_dirs(options.output_dir)

    rng = random.Random(options.seed)

    parameter_specs, mode, max_trials = _load_search_space(options)
    base_config = validate_config(load_config(options.base_config_path))

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
        _serialise_results(results, directories)

    _summarise_completion(results, directories, options)
    _write_log(directories["logs"], "Sweep completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
