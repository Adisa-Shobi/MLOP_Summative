#!/usr/bin/env python3

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
from src.api.endpoints import v1

app = FastAPI()

# Configure CORS
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
