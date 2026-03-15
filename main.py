from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
import time
import uuid
import logging
import cv2
import numpy as np

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join("logs", "tryon.log"),
    format="%(asctime)s %(levelname)s %(message)s"
)

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
    result_path = os.path.join(UPLOAD_DIR, f"{job_id}_result.jpg")

    with open(user_path, "wb") as buffer:
        shutil.copyfileobj(user_image.file, buffer)
    with open(product_path, "wb") as buffer:
        shutil.copyfileobj(product_image.file, buffer)

    # OpenCV Processing
    user_img = cv2.imread(user_path)
    product_img = cv2.imread(product_path)

    # Product image - user image width ku resize
    h, w = user_img.shape[:2]
    product_resized = cv2.resize(product_img, (w // 3, h // 3))

    # Top right corner la overlay
    ph, pw = product_resized.shape[:2]
    result = user_img.copy()
    result[10:10+ph, w-pw-10:w-10] = product_resized

    # Border add pannurom
    cv2.rectangle(result, (w-pw-10, 10), (w-10, 10+ph), (0, 255, 0), 2)

    cv2.imwrite(result_path, result)

    processing_time = round(time.time() - start_time, 3)
    logging.info(f"Job {job_id} processing time: {processing_time}")

    return JSONResponse({
        "status": "success",
        "job_id": job_id,
        "processing_time": processing_time,
        "result_image": f"/result/{job_id}"
    })

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    result_path = os.path.join(UPLOAD_DIR, f"{job_id}_result.jpg")
    if not os.path.exists(result_path):
        return JSONResponse({"error": "Result not found"}, status_code=404)
    return FileResponse(result_path, media_type="image/jpeg")