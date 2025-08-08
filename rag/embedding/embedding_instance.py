from .embedding import Embedding
import asyncio

_embedding_instance = None
_embedding_lock = asyncio.Lock()


async def get_embedding_instance() -> Embedding:
    global _embedding_instance

    if _embedding_instance and _embedding_instance._initialized:
        return _embedding_instance

    async with _embedding_lock:
        if not _embedding_instance:
            _embedding_instance = Embedding()
            await _embedding_instance._ensure_initialized()

    return _embedding_instance
