from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
import time
import uuid
import logging
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector
from PIL import Image

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

# 🔥 DEVICE
device = "cpu"

# 🔥 LOAD CONTROLNET + MODEL
print("⏳ Loading ControlNet Model...")

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose"
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet
)

pipe = pipe.to(device)
pipe.enable_attention_slicing()

# 🔥 POSE DETECTOR
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

print("✅ ControlNet Model Loaded Successfully")


@app.post("/tryon")
async def tryon(
    user_image: UploadFile = File(...),
    product_image: UploadFile = File(...)
):
    print("🔥 TRYON API HIT")

    start_time = time.time()
    job_id = str(uuid.uuid4())

    # validation
    if not user_image.content_type or not user_image.content_type.startswith("image/"):
        return JSONResponse({"error": "Invalid user image"}, status_code=400)

    if not product_image.content_type or not product_image.content_type.startswith("image/"):
        return JSONResponse({"error": "Invalid product image"}, status_code=400)

    logging.info(f"Job {job_id} started")

    user_path = os.path.join(UPLOAD_DIR, f"{job_id}_user.jpg")
    product_path = os.path.join(UPLOAD_DIR, f"{job_id}_product.jpg")
    result_path = os.path.join(UPLOAD_DIR, f"{job_id}_result.png")

    # save images
    with open(user_path, "wb") as buffer:
        shutil.copyfileobj(user_image.file, buffer)

    with open(product_path, "wb") as buffer:
        shutil.copyfileobj(product_image.file, buffer)

    try:
        print("🔥 POSE DETECTION START")

        user_img = Image.open(user_path).convert("RGB")

        # 🔥 pose extraction
        pose_image = openpose(user_img)

        print("🔥 POSE DETECTION DONE")

        prompt = "a full body person wearing a stylish modern outfit"

        print("🔥 CONTROLNET GENERATION START")

        image = pipe(
            prompt,
            image=pose_image,
            num_inference_steps=20
        ).images[0]

        print("🔥 CONTROLNET GENERATION DONE")

        image.save(result_path)

    except Exception as e:
        print("❌ ERROR:", str(e))
        logging.error(f"Job {job_id} failed: {str(e)}")
        return JSONResponse({"error": "Processing failed"}, status_code=500)

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
    result_path = os.path.join(UPLOAD_DIR, f"{job_id}_result.png")

    if not os.path.exists(result_path):
        return JSONResponse({"error": "Result not found"}, status_code=404)

    return FileResponse(result_path, media_type="image/png")