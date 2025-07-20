from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import include_routers
from app.startup import lifespan

app = FastAPI(
    title="Service Diabetes AI",
    description="**Service Diabetes AI** - Dịch vụ AI dự đoán và quản lý tiểu đường",
    version="1.0.0",
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

include_routers(app)
