from app.api.deps import get_current_user
from fastapi import APIRouter, Depends, File, UploadFile

router = APIRouter()


@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...), current_user=Depends(get_current_user)
):
    return {
        "user_id": current_user.id,
        "filename": file.filename,
        "summary": "The uploaded image was recognized as an educational diagram.",
    }
