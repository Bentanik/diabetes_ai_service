from typing import Literal
from app.worker.redis_client import redis_client
from pydantic import BaseModel
import asyncio

from core.cqrs import Mediator

DOCUMENT_QUEUE = "document_jobs"


class DocumentJob(BaseModel):
    file_path: str
    document_id: str
    knowledge_id: str
    title: str
    description: str
    type: Literal["upload_document", "training_document"]


async def add_document_job(job: DocumentJob):
    await redis_client.rpush(DOCUMENT_QUEUE, job.model_dump_json())


async def process_document_upload_job(job: DocumentJob):
    from app.feature.document import ProcessDocumentUploadCommand
    
    command = ProcessDocumentUploadCommand(
        file_path=job.file_path,
        knowledge_id=job.knowledge_id,
        document_id=job.document_id,
        title=job.title,
        description=job.description,
    )
    _ = await Mediator.send(command)


async def process_document_training_job(job: DocumentJob):
    print("Training document")


async def document_worker():
    while True:
        job = await redis_client.blpop(DOCUMENT_QUEUE, timeout=5)
        if job:
            job_json = job[1].decode()
            job_data = DocumentJob.model_validate_json(job_json)
            if job_data.type == "upload_document":
                await process_document_upload_job(job_data)
        else:
            await asyncio.sleep(1)
