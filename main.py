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

        # 2. Pose detection
        shoulders = get_shoulders(user_path)

        if shoulders is None:
            raise Exception("Pose not detected")

        left, right = shoulders
        print("🔥 SHOULDERS:", left, right)

        # 3. Cloth size
        shoulder_width = abs(right[0] - left[0])
        cloth_width = int(shoulder_width * 1.6)
        cloth_height = int(cloth_width * 1.4)

        # 4. Load cloth
        cloth_img = Image.open(product_path).convert("RGBA")
        cloth_resized = cloth_img.resize((cloth_width, cloth_height))
        cloth_np = np.array(cloth_resized)

        # -------------------------------
        # 🔥 WARPING (MAIN LOGIC)
        # -------------------------------
        h_c, w_c = cloth_np.shape[:2]

        src_pts = np.float32([
            [0, 0],
            [w_c, 0],
            [0, h_c],
            [w_c, h_c]
        ])

        # Adjust Y (important fix)
        top_y = int(min(left[1], right[1]) - (cloth_height * 0.25))

        dst_pts = np.float32([
            [left[0], top_y],
            [right[0], top_y],
            [left[0] - 30, top_y + cloth_height],
            [right[0] + 30, top_y + cloth_height]
        ])

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped_cloth = cv2.warpPerspective(
            cloth_np,
            matrix,
            (person_np.shape[1], person_np.shape[0])
        )

        print("🔥 WARPING DONE")

        # -------------------------------
        # 🔥 ALPHA BLENDING
        # -------------------------------
        alpha_mask = warped_cloth[:, :, 3] / 255.0

        for c in range(3):
            person_np[:, :, c] = (
                alpha_mask * warped_cloth[:, :, c] +
                (1 - alpha_mask) * person_np[:, :, c]
            )

        result = Image.fromarray(person_np)

        print("🔥 FINAL OUTPUT READY")

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