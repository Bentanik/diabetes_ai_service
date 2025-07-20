from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    Path,
    Request,
    UploadFile,
    HTTPException,
    Query,
    status,
    Depends,
)

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/haha")
async def haha():
    return {"message": "Hello, World!"}
