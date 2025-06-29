# Hướng dẫn sử dụng LLM đơn giản

AI Service giờ đây chỉ cần **3 biến môi trường** để cấu hình LLM!

## Cấu hình cơ bản

```bash
LLM_BASE_URL=https://openrouter.ai/api/v1    # URL của API
LLM_API_KEY=your_api_key_here                # API key (nếu cần)
LLM_MODEL=deepseek/deepseek-r1-distill-llama-70b:free  # Tên model
```

## Ví dụ cho các provider khác nhau

### 1. OpenRouter (Mặc định)

```bash
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=your_openrouter_api_key
LLM_MODEL=deepseek/deepseek-r1-distill-llama-70b:free
```

### 2. OpenAI

```bash
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_openai_api_key
LLM_MODEL=gpt-3.5-turbo
```

### 3. Localhost/Local APIs

**Text-generation-webui:**

```bash
LLM_BASE_URL=http://localhost:5000/v1
LLM_API_KEY=                              # Để trống
LLM_MODEL=your_model_name
```

**vLLM Server:**

```bash
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=                              # Để trống
LLM_MODEL=meta-llama/Llama-3.3-8B-Instruct
```

**LM Studio:**

```bash
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=                              # Để trống
LLM_MODEL=lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF
```

**Ollama:**

```bash
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=                              # Để trống
LLM_MODEL=llama3.2
```

## Cấu hình tùy chọn

```bash
LLM_TEMPERATURE=0.3      # Độ sáng tạo (0.0 - 1.0)
LLM_MAX_TOKENS=2048      # Số token tối đa trong response
```

## Kiểm tra cấu hình

Gọi API endpoint để xem thông tin hiện tại:

```bash
# Thông tin LLM
curl http://localhost:8000/system/llm-info

# Health check
curl http://localhost:8000/system/health
```

Hoặc trong Python:

```python
from core.llm_client import get_llm

client = get_llm()
info = client.get_provider_info()
print(f"Base URL: {info['base_url']}")
print(f"Model: {info['model']}")
print(f"Has API Key: {info['has_api_key']}")
```

## Lưu ý

- **API Key**: Chỉ cần cho remote APIs (OpenRouter, OpenAI), local servers thường không cần
- **Base URL**: Phải có `/v1` ở cuối để tương thích với OpenAI API format
- **Model Name**: Phải khớp với model có sẵn trên server
- **Backward Compatibility**: Vẫn hỗ trợ `OPENROUTER_API_KEY` từ cấu hình cũ

## So sánh với cấu hình cũ

**Trước đây (phức tạp):**

```bash
LLM_PROVIDER=localhost
LOCALHOST_BASE_URL=http://localhost:8000/v1
LOCALHOST_API_KEY=
LOCALHOST_MODEL=llama-3.3-8b-instruct
```

**Bây giờ (đơn giản):**

```bash
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=llama-3.3-8b-instruct
```
