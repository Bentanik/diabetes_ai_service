import os
from datetime import datetime

# Service Constants
SERVICE_NAME = "Service Diabetes AI"
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "1.0.0")
SERVICE_DESCRIPTION = "AI Service for Diabetes Prediction and Management"
STARTUP_TIME = datetime.now()
