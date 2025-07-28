"""
Document Jobs - Module xử lý các background task liên quan đến tài liệu

File này định nghĩa các task và worker để xử lý các công việc liên quan đến tài liệu như:
- Upload và xử lý tài liệu mới
- Training model với tài liệu
"""

from typing import Literal
import asyncio
from pydantic import BaseModel

from app.worker.redis_client import redis_client
from app.worker.tasks.constants import (
    DOCUMENT_QUEUE,
    QUEUE_POLL_TIMEOUT,
    WORKER_SLEEP_INTERVAL,
)
from core.cqrs import Mediator
from utils import get_logger

logger = get_logger(__name__)


class DocumentJob(BaseModel):
    """
    Model định nghĩa cấu trúc của một document job

    Attributes:
        file_path (str): Đường dẫn đến file tài liệu
        document_id (str): ID của tài liệu
        knowledge_id (str): ID của cơ sở tri thức chứa tài liệu
        title (str): Tiêu đề tài liệu
        description (str): Mô tả tài liệu
        type (str): Loại job - "upload_document" hoặc "training_document"
    """

    file_path: str
    document_id: str
    knowledge_id: str
    title: str
    description: str
    type: Literal["upload_document", "training_document"]


async def add_document_job(job: DocumentJob) -> None:
    """
    Thêm một document job vào queue để xử lý.

    Args:
        job (DocumentJob): Thông tin về job cần xử lý
    """
    logger.info(
        f"Thêm document job mới: type={job.type}, document_id={job.document_id}"
    )
    await redis_client.rpush(DOCUMENT_QUEUE, job.model_dump_json())


async def process_document_upload_job(job: DocumentJob) -> None:
    """
    Xử lý job upload tài liệu.

    Args:
        job (DocumentJob): Thông tin về job upload
    """
    logger.info(f"Xử lý upload tài liệu: document_id={job.document_id}")
    try:
        from app.feature.document import ProcessDocumentUploadCommand

        command = ProcessDocumentUploadCommand(
            file_path=job.file_path,
            knowledge_id=job.knowledge_id,
            document_id=job.document_id,
            title=job.title,
            description=job.description,
        )
        await Mediator.send(command)
        logger.info(f"Hoàn thành xử lý upload tài liệu: document_id={job.document_id}")
    except Exception as e:
        logger.error(
            f"Lỗi xử lý upload tài liệu {job.document_id}: {str(e)}", exc_info=True
        )
        raise


async def process_document_training_job(job: DocumentJob) -> None:
    """
    Xử lý job training với tài liệu.

    Args:
        job (DocumentJob): Thông tin về job training
    """
    logger.info(f"Xử lý training tài liệu: document_id={job.document_id}")
    try:
        # TODO: Implement training logic
        logger.info(f"Hoàn thành training tài liệu: document_id={job.document_id}")
    except Exception as e:
        logger.error(
            f"Lỗi training tài liệu {job.document_id}: {str(e)}", exc_info=True
        )
        raise


async def document_worker() -> None:
    """
    Worker chính để xử lý các document jobs.

    Worker này liên tục kiểm tra queue để lấy và xử lý các job mới.
    Mỗi job sẽ được chuyển đến handler tương ứng dựa vào type.
    """
    logger.info("Document worker started")

    job_handlers = {
        "upload_document": process_document_upload_job,
        "training_document": process_document_training_job,
    }

    while True:
        try:
            job = await redis_client.blpop(DOCUMENT_QUEUE, timeout=QUEUE_POLL_TIMEOUT)
            if job:
                job_json = job[1].decode()
                job_data = DocumentJob.model_validate_json(job_json)

                handler = job_handlers.get(job_data.type)
                if handler:
                    await handler(job_data)
                else:
                    logger.error(
                        f"Không tìm thấy handler cho job type: {job_data.type}"
                    )
            else:
                await asyncio.sleep(WORKER_SLEEP_INTERVAL)
        except Exception as e:
            logger.error(f"Lỗi trong document worker: {str(e)}", exc_info=True)
            await asyncio.sleep(
                WORKER_SLEEP_INTERVAL
            )  # Tránh loop quá nhanh khi có lỗi
