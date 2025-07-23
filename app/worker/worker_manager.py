import asyncio
from app.worker.tasks.document_jobs import document_worker
from utils import get_logger

logger = get_logger(__name__)


class WorkerManager:
    """
    Quản lý các worker
    """

    def __init__(self):
        self.tasks: list[asyncio.Task] = []

    def start_all(self) -> None:
        """
        Khởi động tất cả các worker
        """
        logger.info("Bắt đầu khởi động tất cả các worker...")
        task = asyncio.create_task(document_worker())
        self.tasks.append(task)
        logger.info(f"Worker document_worker được tạo và chạy với task id={id(task)}")
        logger.info("Tất cả các worker đã được khởi động.")

    async def stop_all(self) -> None:
        """
        Dừng tất cả các worker
        """
        logger.info("Bắt đầu dừng tất cả các worker...")
        for task in self.tasks:
            logger.info(f"Hủy task worker với id={id(task)}")
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        logger.info("Tất cả các worker đã được dừng hoàn toàn.")


_worker_manager = WorkerManager()


def get_worker_manager() -> WorkerManager:
    return _worker_manager


def worker_start_all() -> None:
    """
    Khởi động tất cả các worker qua singleton WorkerManager
    """
    logger.info("Gọi worker_start_all() để khởi động worker.")
    _worker_manager.start_all()


async def worker_stop_all() -> None:
    """
    Dừng tất cả các worker qua singleton WorkerManager
    """
    logger.info("Gọi worker_stop_all() để dừng worker.")
    await _worker_manager.stop_all()
