from app.api.deps import get_current_user
from fastapi import APIRouter, Depends, File, UploadFile

router = APIRouter()


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...), current_user=Depends(get_current_user)
):
    return {
        "user_id": current_user.id,
        "filename": file.filename,
        "content_type": file.content_type,
        "stored": True,
    }
