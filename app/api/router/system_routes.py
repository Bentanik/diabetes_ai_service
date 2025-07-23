from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from app.api.schemas import (
    HealthCheckData,
    DatabaseHealth,
    ServiceInfo,
    PingData,
    HealthStatus,
)
from app.database import check_database_health
from core.result.result import Result
from shared import SERVICE_NAME, SERVICE_VERSION
from utils import get_logger
from datetime import datetime
import time

logger = get_logger(__name__)
router = APIRouter(prefix="/system", tags=["System"])

# Track startup time
startup_time = datetime.now()


async def _get_database_health() -> DatabaseHealth:
    """Lấy thông tin database health"""
    try:
        start_time = time.time()
        db_health_raw = await check_database_health()
        response_time = round((time.time() - start_time) * 1000, 2)

        return DatabaseHealth(
            status=(
                HealthStatus.HEALTHY
                if db_health_raw.get("connected", False)
                else HealthStatus.UNHEALTHY
            ),
            connected=db_health_raw.get("connected", False),
            database=db_health_raw.get("database"),
            collections=db_health_raw.get("collections"),
            data_size_mb=db_health_raw.get("data_size_mb"),
            storage_size_mb=db_health_raw.get("storage_size_mb"),
            response_time_ms=response_time,
            error=db_health_raw.get("error"),
        )
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return DatabaseHealth(
            status=HealthStatus.UNHEALTHY,
            connected=False,
            error=str(e),
            response_time_ms=0,
        )


def _get_service_info(db_connected: bool) -> ServiceInfo:
    """Lấy thông tin service"""
    uptime_seconds = int((datetime.now() - startup_time).total_seconds())
    service_status = HealthStatus.HEALTHY if db_connected else HealthStatus.UNHEALTHY

    return ServiceInfo(
        name=SERVICE_NAME,
        version=SERVICE_VERSION,
        status=service_status,
        uptime_seconds=uptime_seconds,
        startup_time=startup_time,
    )


@router.get(
    "/health-check",
    response_model=Result[HealthCheckData],
    responses={
        200: {
            "model": Result[HealthCheckData],
            "description": "Service healthy",
        },
        503: {"model": Result, "description": "Service unhealthy"},
    },
    summary="Health Check",
    description="Kiểm tra sức khỏe tổng thể của service bao gồm database và service status",
)
async def health_check() -> JSONResponse:
    """Kiểm tra sức khỏe service"""
    try:
        # Lấy database health
        database_health = await _get_database_health()

        # Lấy service info
        service_info = _get_service_info(database_health.connected)

        # Xác định overall status
        overall_status = (
            HealthStatus.HEALTHY
            if database_health.connected
            else HealthStatus.UNHEALTHY
        )

        # Tạo health check data
        health_data = HealthCheckData(
            status=overall_status, service=service_info, database=database_health
        )

        if overall_status == HealthStatus.HEALTHY:
            result = Result.success(
                message="Service đang hoạt động bình thường",
                data=health_data,
                code="SERVICE_HEALTHY",
            )
            return result.to_response()
        else:
            # Sử dụng method service_unavailable mới
            result = Result.service_unavailable(
                message="Service không hoạt động",
                code="SERVICE_UNHEALTHY",
                data=health_data,
            )
            return result.to_response()

    except Exception as e:
        logger.error(f"Health check error: {e}")

        result = Result.service_unavailable(
            message="Health check thất bại", code="HEALTH_CHECK_ERROR"
        )
        return result.to_response()


@router.get(
    "/ping",
    response_model=Result[PingData],
    responses={200: {"model": Result[PingData], "description": "Pong response"}},
    summary="Ping Service",
    description="Ping đơn giản để kiểm tra service có phản hồi không",
)
async def ping() -> JSONResponse:
    """Ping đơn giản"""
    try:
        ping_data = PingData(message="pong", service=SERVICE_NAME)

        result = Result.success(
            message="Service đang hoạt động",
            data=ping_data,
            code="SERVICE_HEALTHY",
        )
        return result.to_response()

    except Exception as e:
        logger.error(f"Ping error: {e}")
        result = Result.internal_error(message="Ping thất bại", code="PING_ERROR")
        return result.to_response()
