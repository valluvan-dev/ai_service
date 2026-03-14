from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import time
import uuid
import logging

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO, filename=os.path.join("logs", "tryon.log"), format="%(asctime)s %(levelname)s %(message)s")

@app.post("/tryon")
async def tryon(
    user_image: UploadFile = File(...),
    product_image: UploadFile = File(...)
):
    start_time = time.time()

    job_id = str(uuid.uuid4())

    if not user_image.content_type or not user_image.content_type.startswith("image/"):
        return JSONResponse({"error": "Invalid file type"}, status_code=400)
    if not product_image.content_type or not product_image.content_type.startswith("image/"):
        return JSONResponse({"error": "Invalid file type"}, status_code=400)

    logging.info(f"Job {job_id} started")

    user_path = os.path.join(UPLOAD_DIR, f"{job_id}_user.jpg")
    product_path = os.path.join(UPLOAD_DIR, f"{job_id}_product.jpg")

    with open(user_path, "wb") as buffer:
        shutil.copyfileobj(user_image.file, buffer)

    with open(product_path, "wb") as buffer:
        shutil.copyfileobj(product_image.file, buffer)

    processing_time = round(time.time() - start_time, 3)

    logging.info(f"Processing time: {processing_time}")

    return JSONResponse({
        "status": "success",
        "job_id": job_id,
        "processing_time": processing_time
    })
