from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
import time
import uuid
import logging
import cv2
import numpy as np
import mediapipe as mp

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join("logs", "tryon.log"),
    format="%(asctime)s %(levelname)s %(message)s"
)

# MediaPipe pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

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

    # Images load
    user_img = cv2.imread(user_path)
    product_img = cv2.imread(product_path)
    h, w = user_img.shape[:2]

    # MediaPipe - Pose detect
    rgb = cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Left shoulder + Right shoulder points
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

        # Pixel coordinates calculate
        ls_x = int(left_shoulder.x * w)
        rs_x = int(right_shoulder.x * w)
        ls_y = int(left_shoulder.y * h)
        lh_y = int(left_hip.y * h)

        # Chest region width + height
        chest_w = abs(ls_x - rs_x)
        chest_h = int((lh_y - ls_y) * 0.8)

        # Product image resize - chest size ku
        product_resized = cv2.resize(product_img, (chest_w, chest_h))

        # Overlay position
        x_start = min(ls_x, rs_x)
        y_start = ls_y

        # Boundary check
        x_end = min(x_start + chest_w, w)
        y_end = min(y_start + chest_h, h)

        actual_w = x_end - x_start
        actual_h = y_end - y_start

        # Overlay pannurom
        result = user_img.copy()
        result[y_start:y_end, x_start:x_end] = product_resized[:actual_h, :actual_w]

        # Green border
        cv2.rectangle(result, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    else:
        # Pose detect aagala - corner la போடுvooom
        logging.warning(f"Job {job_id} - Pose not detected, using fallback")
        result = user_img.copy()
        ph, pw = product_img.shape[:2]
        product_resized = cv2.resize(product_img, (w // 3, h // 3))
        ph, pw = product_resized.shape[:2]
        result[10:10+ph, w-pw-10:w-10] = product_resized

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