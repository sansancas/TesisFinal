import io
from typing import List

import numpy as np
import pandas as pd
import pyedflib
from fastapi import HTTPException

from classes.models import ChannelSummary, SeizureEvent


def _parse_edf_file(edf_path: str):
    try:
        reader = pyedflib.EdfReader(edf_path)
    except Exception as exc:  # pragma: no cover - error propagation
        raise HTTPException(status_code=400, detail=f"No se pudo leer el archivo EDF: {exc}") from exc

    try:
        n_signals = reader.signals_in_file
        if n_signals <= 0:
            raise HTTPException(status_code=400, detail="El archivo EDF no contiene señales.")

        labels = reader.getSignalLabels()
        sample_frequencies = [reader.getSampleFrequency(i) for i in range(n_signals)]
        raw_signals = [reader.readSignal(i) for i in range(n_signals)]
        sample_counts = reader.getNSamples()
    finally:
        reader.close()

    signal_bundle = []
    for label, freq, signal, count in zip(labels, sample_frequencies, raw_signals, sample_counts):
        normalized_label = str(label).strip() or f"channel_{len(signal_bundle)}"
        normalized_label_lower = normalized_label.lower()
        numeric_freq = float(freq) if freq is not None else 0.0

        if numeric_freq <= 0 or count == 0 or normalized_label_lower == "edf annotations":
            continue

        np_signal = np.asarray(signal, dtype=float)
        if np_signal.size == 0:
            continue

        signal_bundle.append((normalized_label, numeric_freq, np_signal))

    if not signal_bundle:
        raise HTTPException(status_code=400, detail="No se encontraron canales válidos en el EDF.")

    freq_set = {round(freq, 6) for _, freq, _ in signal_bundle}
    length_set = {signal.size for _, _, signal in signal_bundle}

    aligned_sampling = len(freq_set) == 1 and len(length_set) == 1
    duration_seconds = float(max(signal.size / freq for _, freq, signal in signal_bundle))

    channels: List[ChannelSummary] = []
    for label, freq, signal in signal_bundle:
        signal = signal.astype(float)
        sample_count = int(signal.size)
        channel_duration = float(sample_count / freq) if freq else 0.0
        amplitude_min = float(np.min(signal))
        amplitude_max = float(np.max(signal))
        amplitude_mean = float(np.mean(signal))
        amplitude_std = float(np.std(signal))
        channel_payload = ChannelSummary(
            label=label,
            sample_frequency=float(freq),
            sample_count=sample_count,
            duration_seconds=channel_duration,
            amplitude_min=amplitude_min,
            amplitude_max=amplitude_max,
            amplitude_mean=amplitude_mean,
            amplitude_std=amplitude_std,
        )
        channels.append(channel_payload)

    return channels, aligned_sampling, duration_seconds


def _parse_events_csv(csv_bytes: bytes) -> List[SeizureEvent]:
    try:
        text_stream = io.StringIO(csv_bytes.decode("utf-8-sig"))
        df = pd.read_csv(text_stream, comment="#")
    except Exception as exc:  # pragma: no cover - error propagation
        raise HTTPException(status_code=400, detail=f"No se pudo leer el CSV: {exc}") from exc

    if df.empty:
        return []

    df.columns = [str(col).strip() for col in df.columns]

    required_columns = {"start_time", "stop_time", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"El CSV carece de columnas obligatorias: {', '.join(sorted(missing))}",
        )

    df = df.dropna(subset=["start_time", "stop_time", "label"])
    if df.empty:
        return []

    df["label"] = df["label"].astype(str).str.strip()
    seizure_rows = df[df["label"].str.lower() == "seiz"]
    if seizure_rows.empty:
        return []

    events: List[SeizureEvent] = []
    for _, row in seizure_rows.iterrows():
        channel_value = row.get("channel")
        channel = None if pd.isna(channel_value) else str(channel_value).strip()
        confidence_value = row.get("confidence")
        confidence = None if pd.isna(confidence_value) else float(confidence_value)
        events.append(
            SeizureEvent(
                channel=channel,
                start_time=float(row["start_time"]),
                stop_time=float(row["stop_time"]),
                label=str(row["label"]),
                confidence=confidence,
            )
        )

    return events
