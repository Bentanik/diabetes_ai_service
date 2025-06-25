from fastapi import FastAPI
from api.router import router as api_router

app = FastAPI(title="AI Service - CarePlan Generator", version="1.0.0")

app.include_router(api_router)
