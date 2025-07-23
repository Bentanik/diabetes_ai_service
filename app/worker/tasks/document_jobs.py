from app.worker.redis_client import redis_client
from pydantic import BaseModel
import asyncio

DOCUMENT_QUEUE = "document_jobs"


class DocumentJob(BaseModel):
    knowledge_id: str
    title: str
    description: str
    file_path: str


async def add_document_job(job: DocumentJob):
    await redis_client.rpush(DOCUMENT_QUEUE, job.model_dump_json())


async def process_document_job(job: DocumentJob):
    print(f"Processing document job: {job.title}")
    # Thực hiện xử lý tài liệu, ví dụ:
    await asyncio.sleep(130)


async def document_worker():
    while True:
        job = await redis_client.blpop(DOCUMENT_QUEUE, timeout=5)
        if job:
            job_json = job[1].decode()
            job_data = DocumentJob.model_validate_json(job_json)
            await process_document_job(job_data)
        else:
            await asyncio.sleep(1)
