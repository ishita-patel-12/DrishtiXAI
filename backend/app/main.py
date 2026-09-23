import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.deps import init_db
from app.routers import analyze, cases, reports, review, system

# Create database tables if not existing
init_db()

app = FastAPI(
    title="DrishtiXAI Backend API",
    description="Explainable AI System for Diabetic Retinopathy Screening (SIH26038)",
    version="0.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs static directory (/backend/outputs -> /backend/outputs)
app.mount("/backend/outputs", StaticFiles(directory=str(settings.OUTPUT_DIR)), name="outputs")

# Include Routers
app.include_router(analyze.router)
app.include_router(cases.router)
app.include_router(reports.router)
app.include_router(review.router)
app.include_router(system.router)

# Serve Frontend SPA
@app.get("/")
async def serve_index():
    html_file = settings.FRONTEND_DIR / "drishtixai.html"
    if html_file.exists():
        return FileResponse(html_file)
    return {"message": "DrishtiXAI API Backend is running. Frontend file not found."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
