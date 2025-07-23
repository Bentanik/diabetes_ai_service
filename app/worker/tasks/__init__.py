from app.worker.redis_client import redis_client
import asyncio

DOCUMENT_QUEUE = "document_jobs"


async def add_document_job(job_data: dict):
    await redis_client.rpush(DOCUMENT_QUEUE, str(job_data))


async def process_document_job(job_data: str):
    print(f"Processing document job: {job_data}")
    await asyncio.sleep(3)


async def document_worker():
    while True:
        job = await redis_client.blpop(DOCUMENT_QUEUE, timeout=5)
        if job:
            job_data = job[1].decode()
            await process_document_job(job_data)
        else:
            await asyncio.sleep(1)
