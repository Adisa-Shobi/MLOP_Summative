#!/usr/bin/env python3

from fastapi import FastAPI
import uvicorn
import os
import sys
from src.api.endpoints import v1

app = FastAPI()

app.include_router(v1)

if __name__ == "__main__":
    # Run the FastAPI application with Uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",  # Assuming this code is in main.py
        host="0.0.0.0",
        port=port,
        reload=True,  # Enable auto-reload during development
        workers=1
    )
