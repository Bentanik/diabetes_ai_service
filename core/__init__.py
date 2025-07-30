"""
Package core - Tầng Core trong Clean Architecture.

Chứa các components cốt lõi của ứng dụng:
- llm: Module quản lý language models
- cqrs: Command Query Responsibility Segregation pattern
- result: Result pattern cho error handling

Nguyên tắc:
- Core không phụ thuộc vào bất kỳ tầng nào khác
- Chứa business logic và domain models
- Là trung tâm của toàn bộ kiến trúc
"""

# Không cần import gì cả, các module khác sẽ import trực tiếp từ core.llm
