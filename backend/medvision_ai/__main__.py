"""Run the API on loopback only; access logs are disabled to avoid content leakage."""

import uvicorn

import os

uvicorn.run(
    "medvision_ai.api:app",
    host="127.0.0.1",
    port=int(os.environ.get("MEDVISION_API_PORT", "8765")),
    access_log=False,
)
