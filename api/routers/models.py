import os
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from classes.models import EDFAnalysisResponse, NetworkEvaluationResponse, SeizureEvent
from processing.files import _parse_edf_file, _parse_events_csv
from processing.network_eval import evaluate_network, _run_evaluation

router = APIRouter(
    prefix="/models",
    tags=["Models"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def all_atm():
    return {"message": "models"}

@router.post("/analyze", response_model=EDFAnalysisResponse, status_code=status.HTTP_200_OK)
async def analyze_edf(
    edf_file: UploadFile = File(...),
    csv_file: Optional[UploadFile] = File(None),
):
    """Procesa un EDF y un CSV opcional para obtener amplitudes y eventos de convulsión."""

    if not edf_file.filename:
        raise HTTPException(status_code=400, detail="Se requiere un archivo EDF válido.")

    edf_bytes = await edf_file.read()
    if not edf_bytes:
        raise HTTPException(status_code=400, detail="El archivo EDF está vacío.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_edf:
        tmp_edf.write(edf_bytes)
        tmp_edf.flush()
        edf_path = tmp_edf.name
        print(f"Archivo EDF temporal creado en: {edf_path}")

    try:
        channels, aligned, duration = _parse_edf_file(edf_path)
    finally:
        os.unlink(edf_path)

    events: List[SeizureEvent] = []
    if csv_file is not None and csv_file.filename:
        csv_bytes = await csv_file.read()
        if csv_bytes:
            events = _parse_events_csv(csv_bytes)

    return EDFAnalysisResponse(
        aligned_sampling=aligned,
        duration_seconds=duration,
        channel_count=len(channels),
        channels=channels,
        events=events,
    )

@router.post(
    "/eval_network_tf",
    response_model=NetworkEvaluationResponse,
    status_code=status.HTTP_200_OK,
)
async def eval_network_tf(
    edf_file: UploadFile = File(...),
    csv_file: Optional[UploadFile] = File(None),
) -> NetworkEvaluationResponse:
    """Evalúa el modelo TensorFlow sobre un EDF y un CSV opcional con etiquetas."""

    return await _run_evaluation("tf", edf_file, csv_file)


@router.post(
    "/eval_network_pt",
    response_model=NetworkEvaluationResponse,
    status_code=status.HTTP_200_OK,
)
async def eval_network_pt(
    edf_file: UploadFile = File(...),
    csv_file: Optional[UploadFile] = File(None),
) -> NetworkEvaluationResponse:
    """Evalúa el modelo PyTorch sobre un EDF y un CSV opcional con etiquetas."""

    return await _run_evaluation("pt", edf_file, csv_file)