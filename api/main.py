import os
import logging
from fastapi import FastAPI, APIRouter
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from routers import models

import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/openapi.json",
)

api_router = APIRouter(prefix="/api")
api_router.include_router(models.router)

app.include_router(api_router)

_cors_origins_env = os.getenv("CORS_ALLOW_ORIGINS")
if _cors_origins_env:
    _cors_origins = [origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()]
else:
    _cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse('api/index/index.html')


# css for the index.html
@app.get("/css/bootstrap.css", response_class=FileResponse)
async def root():
    return FileResponse('api/index/css/bootstrap.css')


@app.get("/css/style.css", response_class=FileResponse)
async def root():
    return FileResponse('api/index/css/style.css')


@app.get("/css/responsive.css", response_class=FileResponse)
async def root():
    return FileResponse('api/index/css/responsive.css')


@app.get("/js/jquery-3.4.1.min.js", response_class=FileResponse)
async def root():
    return FileResponse('api/index/js/jquery-3.4.1.min.js')


@app.get("/js/bootstrap.js", response_class=FileResponse)
async def root():
    return FileResponse('api/index/js/bootstrap.js')


@app.get("/images/camping.svg", response_class=FileResponse)
async def root():
    return FileResponse('api/index/images/camping.svg')


@app.get("/images/hero-bg2.png", response_class=FileResponse)
async def root():
    return FileResponse('api/index/images/hero-bg2.png')


@app.get("/css/style.css.map", response_class=FileResponse)
async def root():
    return FileResponse('api/index/css/style.css.map')
