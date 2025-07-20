from fastapi import FastAPI
from app.api.router import rag_router
from app.api.router import system_router


def include_routers(app: FastAPI):
    """
    Cấu hình và đăng ký tất cả các router cho FastAPI application

    Args:
        app (FastAPI): Instance của FastAPI application

    Description:
        Function này tập trung việc đăng ký các router để:
        - Dễ quản lý các endpoint
        - Tách biệt logic routing
        - Dễ thêm/xóa router mới
    """
    app.include_router(rag_router)
    app.include_router(system_router)
