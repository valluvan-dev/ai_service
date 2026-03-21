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
# 🔥 SEGMENTATION FUNCTION
# -------------------------------
def segment_person(image_path):
    input_image = Image.open(image_path).convert("RGBA")
    output = remove(input_image)
    return output


# -------------------------------
# 🔥 CLOTH PREP FUNCTION
# -------------------------------
def prepare_cloth(cloth_path, target_size):
    cloth = Image.open(cloth_path).convert("RGBA")
    cloth = cloth.resize(target_size)
    return cloth


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

    # Save images
    with open(user_path, "wb") as f:
        shutil.copyfileobj(user_image.file, f)

    with open(product_path, "wb") as f:
        shutil.copyfileobj(product_image.file, f)

    try:
        print("🔥 SEGMENTATION START")

        # 1. Segment person
        person_img = segment_person(user_path)

        print("🔥 SEGMENTATION DONE")

        # 2. Prepare cloth
        cloth_img = prepare_cloth(product_path, person_img.size)

        print("🔥 CLOTH PREPARED")

        # 3. Convert to numpy
        person_np = np.array(person_img)
        cloth_np = np.array(cloth_img)

        result_np = person_np.copy()

        h, w = person_np.shape[:2]

        # 4. Chest region (approx)
        x1 = int(w * 0.25)
        y1 = int(h * 0.30)
        x2 = int(w * 0.75)
        y2 = int(h * 0.65)

        cloth_resized = np.array(
            cloth_img.resize((x2 - x1, y2 - y1))
        )

        # 5. Overlay cloth
        result_np[y1:y2, x1:x2] = cloth_resized

        result = Image.fromarray(result_np)

        print("🔥 TRY-ON DONE")

        result.save(result_path)

    except Exception as e:
        print("❌ ERROR:", str(e))
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