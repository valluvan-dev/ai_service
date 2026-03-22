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
from diffusers import StableDiffusionImg2ImgPipeline

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
# 🔥 MEDIAPIPE
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

        return (int(left.x * w), int(left.y * h)), (int(right.x * w), int(right.y * h))


# -------------------------------
# 🔥 SEGMENTATION
# -------------------------------
def segment_person(image_path):
    input_image = Image.open(image_path).convert("RGBA")
    return remove(input_image)


# -------------------------------
# 🔥 LOAD AI MODEL
# -------------------------------
print("⏳ Loading AI Model...")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)
pipe = pipe.to("cpu")
pipe.enable_attention_slicing()
print("✅ AI Model Loaded")


# -------------------------------
# 🔥 TRYON API
# -------------------------------
@app.post("/tryon")
async def tryon(user_image: UploadFile = File(...), product_image: UploadFile = File(...)):

    start_time = time.time()
    job_id = str(uuid.uuid4())

    user_path = os.path.join(UPLOAD_DIR, f"{job_id}_user.png")
    product_path = os.path.join(UPLOAD_DIR, f"{job_id}_product.png")
    result_path = os.path.join(UPLOAD_DIR, f"{job_id}_result.png")

    with open(user_path, "wb") as f:
        shutil.copyfileobj(user_image.file, f)

    with open(product_path, "wb") as f:
        shutil.copyfileobj(product_image.file, f)

    try:
        # -------------------------------
        # 1. SEGMENT PERSON
        # -------------------------------
        person_img = segment_person(user_path)

        # 🔥 IMPORTANT FIX
        person_np = np.array(person_img)
        person_rgb = person_np[:, :, :3]   # remove alpha

        # -------------------------------
        # 2. POSE
        # -------------------------------
        shoulders = get_shoulders(user_path)
        if shoulders is None:
            raise Exception("Pose not detected")

        left, right = shoulders

        # -------------------------------
        # 3. CLOTH
        # -------------------------------
        shoulder_width = abs(right[0] - left[0])
        cloth_width = int(shoulder_width * 1.6)
        cloth_height = int(cloth_width * 1.4)

        cloth_img = Image.open(product_path).convert("RGBA")
        cloth_np = np.array(cloth_img.resize((cloth_width, cloth_height)))

        # -------------------------------
        # 4. WARP
        # -------------------------------
        h_c, w_c = cloth_np.shape[:2]

        src_pts = np.float32([
            [0, 0], [w_c, 0],
            [0, h_c], [w_c, h_c]
        ])

        top_y = int(min(left[1], right[1]) - (cloth_height * 0.35))

        dst_pts = np.float32([
            [left[0], top_y],
            [right[0], top_y],
            [left[0] - 30, top_y + cloth_height],
            [right[0] + 30, top_y + cloth_height]
        ])

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped = cv2.warpPerspective(
            cloth_np,
            matrix,
            (person_rgb.shape[1], person_rgb.shape[0])
        )

        # -------------------------------
        # 5. BLEND (FIXED)
        # -------------------------------
        alpha = warped[:, :, 3] / 255.0
        alpha = np.where(alpha > 0.1, alpha, 0)

        for c in range(3):
            person_rgb[:, :, c] = (
                alpha * warped[:, :, c] +
                (1 - alpha) * person_rgb[:, :, c]
            )

        # -------------------------------
        # 6. AI REFINEMENT (SAFE)
        # -------------------------------
        input_img = Image.fromarray(person_rgb.astype(np.uint8))

        ai_result = pipe(
            prompt="realistic photo of a person wearing a shirt",
            image=input_img,
            strength=0.3,   # 🔥 reduced (important)
            guidance_scale=7
        ).images[0]

        ai_result.save(result_path)

    except Exception as e:
        print("❌ ERROR:", str(e))
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({
        "status": "success",
        "job_id": job_id,
        "processing_time": round(time.time() - start_time, 3),
        "result_image": f"/result/{job_id}"
    })


# -------------------------------
# RESULT
# -------------------------------
@app.get("/result/{job_id}")
async def get_result(job_id: str):
    path = os.path.join(UPLOAD_DIR, f"{job_id}_result.png")
    if not os.path.exists(path):
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(path)