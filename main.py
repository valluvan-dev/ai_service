from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
import time
import uuid
import logging
from PIL import Image
import numpy as np
from rembg import remove
import cv2
import mediapipe as mp

print("🚀 MAIN FILE LOADED")

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join("logs", "tryon.log"),
    format="%(asctime)s %(levelname)s %(message)s"
)

# -------------------------------
# 🔥 MEDIAPIPE SETUP
# -------------------------------
mp_pose = mp.solutions.pose


def get_shoulders(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            return None

        h, w, _ = image.shape

        left = results.pose_landmarks.landmark[11]
        right = results.pose_landmarks.landmark[12]

        left_shoulder = (int(left.x * w), int(left.y * h))
        right_shoulder = (int(right.x * w), int(right.y * h))

        return left_shoulder, right_shoulder


# -------------------------------
# 🔥 SEGMENTATION
# -------------------------------
def segment_person(image_path):
    input_image = Image.open(image_path).convert("RGBA")
    output = remove(input_image)
    return output


# -------------------------------
# 🔥 TRYON API
# -------------------------------
@app.post("/tryon")
async def tryon(
    user_image: UploadFile = File(...),
    product_image: UploadFile = File(...)
):
    print("🔥 TRYON API HIT")

    start_time = time.time()
    job_id = str(uuid.uuid4())

    if not user_image.content_type.startswith("image/"):
        return JSONResponse({"error": "Invalid user image"}, status_code=400)

    if not product_image.content_type.startswith("image/"):
        return JSONResponse({"error": "Invalid product image"}, status_code=400)

    user_path = os.path.join(UPLOAD_DIR, f"{job_id}_user.png")
    product_path = os.path.join(UPLOAD_DIR, f"{job_id}_product.png")
    result_path = os.path.join(UPLOAD_DIR, f"{job_id}_result.png")

    # Save files
    with open(user_path, "wb") as f:
        shutil.copyfileobj(user_image.file, f)

    with open(product_path, "wb") as f:
        shutil.copyfileobj(product_image.file, f)

    try:
        print("🔥 SEGMENTATION START")

        # 1. Segment person
        person_img = segment_person(user_path)
        person_np = np.array(person_img)

        print("🔥 SEGMENTATION DONE")

        # 2. Detect shoulders
        print("🔥 POSE DETECTION START")
        shoulders = get_shoulders(user_path)

        if shoulders is None:
            raise Exception("Pose not detected")

        left, right = shoulders
        print("🔥 POSE DETECTED:", left, right)

        # 3. Calculate cloth size
        shoulder_width = abs(right[0] - left[0])

        cloth_width = int(shoulder_width * 1.3)
        cloth_height = int(cloth_width * 1.4)

        # 4. Load cloth
        cloth_img = Image.open(product_path).convert("RGBA")
        cloth_resized = cloth_img.resize((cloth_width, cloth_height))
        cloth_np = np.array(cloth_resized)

        print("🔥 CLOTH RESIZED")

        # 5. Position calculation
        center_x = int((left[0] + right[0]) / 2)
        top_y = min(left[1], right[1])

        x1 = int(center_x - cloth_width / 2)
        y1 = int(top_y)

        x2 = x1 + cloth_width
        y2 = y1 + cloth_height

        # 6. Boundary check
        h, w = person_np.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        cloth_crop = cloth_np[0:(y2 - y1), 0:(x2 - x1)]

        # 7. Overlay with alpha blending
        result_np = person_np.copy()

        alpha = 0.7

        region = result_np[y1:y2, x1:x2]

        blended = (
            alpha * cloth_crop + (1 - alpha) * region
        ).astype(np.uint8)

        result_np[y1:y2, x1:x2] = blended

        result = Image.fromarray(result_np)

        print("🔥 FINAL TRY-ON DONE")

        result.save(result_path)

    except Exception as e:
        print("❌ ERROR:", str(e))
        logging.error(str(e))
        return JSONResponse({"error": "Processing failed"}, status_code=500)

    processing_time = round(time.time() - start_time, 3)

    return JSONResponse({
        "status": "success",
        "job_id": job_id,
        "processing_time": processing_time,
        "result_image": f"/result/{job_id}"
    })


# -------------------------------
# 🔥 RESULT API
# -------------------------------
@app.get("/result/{job_id}")
async def get_result(job_id: str):
    result_path = os.path.join(UPLOAD_DIR, f"{job_id}_result.png")

    if not os.path.exists(result_path):
        return JSONResponse({"error": "Result not found"}, status_code=404)

    return FileResponse(result_path, media_type="image/png")