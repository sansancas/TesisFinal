from typing import List, Optional

from pydantic import BaseModel


class ChannelSummary(BaseModel):
    label: str
    sample_frequency: float
    sample_count: int
    duration_seconds: float
    amplitude_min: float
    amplitude_max: float
    amplitude_mean: float
    amplitude_std: float


class SeizureEvent(BaseModel):
    channel: Optional[str] = None
    start_time: float
    stop_time: float
    label: str
    confidence: Optional[float] = None


class EDFAnalysisResponse(BaseModel):
    aligned_sampling: bool
    duration_seconds: float
    channel_count: int
    channels: List[ChannelSummary]
    events: List[SeizureEvent]


class PredictionInterval(BaseModel):
    start_time: float
    stop_time: float
    score: float
    count: int


class EvaluationSummary(BaseModel):
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    confusion_matrix: List[List[float]]


class NetworkEvaluationResponse(BaseModel):
    model_type: str
    model_architecture: str
    window_sec: float
    hop_sec: float
    threshold: float
    events: List[SeizureEvent]
    metrics: Optional[EvaluationSummary] = None