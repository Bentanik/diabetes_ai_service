from fastapi import APIRouter, HTTPException, status
from app.api.schemas import (
    SuccessResponse,
    ErrorResponse,
    HealthCheckData,
    DatabaseHealth,
    ServiceInfo,
    PingData,
    HealthStatus,
)
from app.database import check_database_health
from shared import SERVICE_NAME, SERVICE_VERSION
from utils import get_logger
from datetime import datetime
import time

logger = get_logger(__name__)
router = APIRouter(prefix="/system", tags=["System"])

# Track startup time
startup_time = datetime.now()


async def _get_database_health() -> DatabaseHealth:
    """Helper function để lấy thông tin database health"""
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
            status=HealthStatus.UNHEALTHY, connected=False, error=str(e)
        )


def _get_service_info(db_connected: bool) -> ServiceInfo:
    """Helper function để lấy thông tin service"""
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
    response_model=SuccessResponse[HealthCheckData],
    responses={
        200: {
            "model": SuccessResponse[HealthCheckData],
            "description": "Service healthy",
        },
        503: {"model": ErrorResponse, "description": "Service unhealthy"},
    },
    summary="Health Check",
    description="Kiểm tra sức khỏe tổng thể của service bao gồm database và service status",
)
async def health_check():
    """Kiểm tra sức khỏe service"""
    try:
        # Get database health
        database_health = await _get_database_health()

        # Get service info
        service_info = _get_service_info(database_health.connected)

        # Determine overall status
        overall_status = (
            HealthStatus.HEALTHY
            if database_health.connected
            else HealthStatus.UNHEALTHY
        )

        # Create health check data
        health_data = HealthCheckData(
            status=overall_status, service=service_info, database=database_health
        )

        if overall_status == HealthStatus.HEALTHY:
            return SuccessResponse(
                message="Service đang hoạt động bình thường",
                data=health_data,
                code="SERVICE_HEALTHY",
                isSuccess=True,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=ErrorResponse(
                    detail="Service không hoạt động",
                    errorCode="SERVICE_UNHEALTHY",
                    status=503,
                    title="Service Unavailable",
                ).model_dump(),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check error: {e}")
        # ✅ FIX: Sửa cách tạo ErrorResponse
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorResponse(
                detail=str(e),
                errorCode="HEALTH_CHECK_ERROR",
                status=503,
                title="Health Check Failed",
            ).model_dump(),
        )


@router.get(
    "/ping",
    response_model=SuccessResponse[PingData],
    responses={
        200: {"model": SuccessResponse[PingData], "description": "Pong response"}
    },
    summary="Ping Service",
    description="Ping đơn giản để kiểm tra service có phản hồi không",
)
async def ping():
    """Ping đơn giản"""
    ping_data = PingData(message="pong", service=SERVICE_NAME)

    return SuccessResponse(
        isSuccess=True,
        code="SERVICE_HEALTHY",
        message="Service đang hoạt động",
        data=ping_data,
    )
