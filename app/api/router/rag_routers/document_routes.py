from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.feature.document import CreateDocumentCommand
from utils import get_logger
from core.cqrs import Mediator
from utils import get_scorer

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=None,
    summary="Tạo tài liệu mới",
    description="Tạo mới tài liệu trong hệ thống.",
)
async def create_document(
    file: UploadFile = File(...),
    knowledge_id: str = Form(...),
    title: str = Form(...),
    description: str = Form(...),
) -> JSONResponse:
    logger.info(f"Tạo tài liệu mới: {title}")
    try:
        doc_req = CreateDocumentCommand(
            file=file, knowledge_id=knowledge_id, title=title, description=description
        )
        result = await Mediator.send(doc_req)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi tạo tài liệu: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Tạo tài liệu thất bại")


@router.get("/test")
async def test_embedding_model():
    test_texts = [
        "Bệnh tiểu đường type 2 là bệnh mãn tính. Bệnh nhân có triệu chứng khát nước và đi tiểu nhiều.",
        "Diabetes mellitus affects glucose metabolism. Treatment includes insulin therapy.",
        "Hôm nay trời đẹp, tôi đi chơi công viên với bạn bè.",
        "The patient presented with elevated HbA1c levels.",
        "Công ty phần mềm phát triển ứng dụng mobile.",
        "Biến chứng tiểu đường có thể ảnh hưởng nghiêm trọng đến sức khỏe.",
    ]

    scorer = get_scorer()
    for i, text in enumerate(test_texts, 1):
        print(f"\nText {i}: {text[:50]}...")
        analysis = scorer.get_detailed_analysis(text)
        print(f"Final Score: {analysis['final_score']} ({analysis['relevance_level']})")
        print(
            f"Semantic: {analysis['semantic_score']}, Keyword: {analysis['keyword_score']}"
        )
    return {"message": "Hello, World!"}
