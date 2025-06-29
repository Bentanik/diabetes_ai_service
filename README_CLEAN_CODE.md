# Clean Code Refactoring Documentation

## Overview

Dự án đã được refactor để có clean code và architecture tốt hơn với các cải tiến sau:

## 🏗️ Cấu trúc mới

### 1. Configuration Management

- **`config/settings.py`**: Quản lý tập trung tất cả configuration
- Sử dụng Pydantic Settings với environment variables
- Type-safe configuration

### 2. Exception Handling

- **`core/exceptions.py`**: Custom exception classes
- Hierarchical exception structure
- Better error messages and error codes

### 3. Logging System

- **`core/logging_config.py`**: Centralized logging configuration
- Structured logging với timestamps
- Configurable log levels và file output

### 4. Interface/Protocol Design

- **`core/interfaces.py`**: Abstract interfaces cho services
- Protocol-based design cho testability
- Dependency injection support

### 5. Service Layer

- **`services/`**: Business logic layer
- **`services/care_plan_service.py`**: Care plan generation
- **`services/measurement_service.py`**: Measurement analysis
- Clean separation of concerns

### 6. Enhanced LLM Client

- **`core/llm_client.py`**: Improved LLM client với error handling
- Configuration-driven setup
- Singleton pattern với factory functions

## 🔧 Cải tiến chính

### 1. Error Handling

```python
# Trước
try:
    result = await some_function()
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

# Sau
try:
    result = await service.process_request(request)
except BaseServiceException as e:
    raise HTTPException(
        status_code=400,
        detail={"error": e.error_code, "message": e.message}
    )
```

### 2. Configuration Management

```python
# Trước
openai_api_key = os.getenv("OPENROUTER_API_KEY")
model = "meta-llama/llama-3.3-8b-instruct:free"

# Sau
from config.settings import settings
api_key = settings.openrouter_api_key
model = settings.default_model
```

### 3. Logging

```python
# Trước
print(f"Processing request for patient {patient_id}")

# Sau
from core.logging_config import get_logger
logger = get_logger(__name__)
logger.info(f"Processing request for patient {patient_id}")
```

### 4. Service Layer

```python
# Trước (direct function call)
result = await generate_careplan_measurements(request)

# Sau (service pattern)
service = get_care_plan_service()
result = await service.generate_care_plan(request)
```

## 📝 Environment Variables

Tạo file `.env` với các biến sau:

```bash
# Required
OPENROUTER_API_KEY=your_api_key_here

# Optional (có default values)
LOG_LEVEL=INFO
DEFAULT_MODEL=meta-llama/llama-3.3-8b-instruct:free
DEFAULT_TEMPERATURE=0.3
MAX_REASON_LENGTH=150
MAX_FEEDBACK_LENGTH=250
```

## 🚀 Cách chạy ứng dụng

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your values

# Run the application
uvicorn main:app --reload
```

## 📊 API Endpoints

### Sau khi refactor, các endpoint được thay đổi:

- **Health Check**: `GET /health`
- **Care Plan**: `POST /api/v1/careplan/generate`
- **Measurement Analysis**: `POST /api/v1/analyze/analyze-measurement-note`
- **API Docs**: `GET /docs`

## 🧪 Testing

```bash
# Run tests
pytest

# Run với coverage
pytest --cov=.
```

## 📁 File Structure Mới

```
aiservice/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration management
├── core/
│   ├── __init__.py
│   ├── exceptions.py        # Custom exceptions
│   ├── interfaces.py        # Abstract interfaces
│   ├── llm_client.py       # Enhanced LLM client
│   └── logging_config.py   # Logging setup
├── services/
│   ├── __init__.py
│   ├── care_plan_service.py    # Care plan business logic
│   └── measurement_service.py   # Measurement analysis logic
├── api/                     # API routes (updated)
├── features/               # Legacy code (deprecated)
├── models/                 # Pydantic models
├── prompts/               # Prompt templates
├── utils/                 # Utility functions (enhanced)
├── constants/             # Constants and schemas
└── main.py               # Application entry point (updated)
```

## 🎯 Benefits của Clean Code Refactoring

1. **Maintainability**: Code dễ đọc và maintain hơn
2. **Testability**: Dependency injection và interfaces giúp testing dễ hơn
3. **Scalability**: Service layer có thể mở rộng dễ dàng
4. **Error Handling**: Comprehensive error handling với structured errors
5. **Configuration**: Centralized configuration management
6. **Logging**: Structured logging để debug và monitor
7. **Type Safety**: Better type hints và validation
8. **Separation of Concerns**: Clear separation giữa layers

## 🔄 Migration Notes

- Legacy functions vẫn hoạt động nhưng có deprecation warnings
- API endpoints có prefix `/api/v1` mới
- Environment variables được quản lý bởi Pydantic Settings
- Error responses có format mới với error codes

## 📋 TODO / Future Improvements

1. Add comprehensive unit tests
2. Implement async database layer
3. Add monitoring và metrics
4. Implement caching layer
5. Add API rate limiting
6. Enhance chatbot service implementation
7. Add request/response validation middleware
8. Implement proper authentication/authorization
