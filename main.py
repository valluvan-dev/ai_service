from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import os, uuid, base64, httpx
from PIL import Image
import io

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

FAL_KEY = "516e54ee-aef6-400c-b0cd-7245254c81c9:90081abadaaf59fb3ccac6b35cf8af75"  # ← Inga mattum change pannu
FAL_URL = "https://fal.run/fal-ai/fashn/tryon/v1.6"


@app.post("/tryon")
async def tryon(user_image: UploadFile = File(...), product_image: UploadFile = File(...)):
    job_id = str(uuid.uuid4())

    # Read images
    user_bytes = await user_image.read()
    product_bytes = await product_image.read()

    # Resize images
    user_img = Image.open(io.BytesIO(user_bytes)).convert("RGB").resize((512, 768))
    product_img = Image.open(io.BytesIO(product_bytes)).convert("RGB").resize((512, 768))

    # Convert to base64
    def to_base64(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                FAL_URL,
                headers={
                    "Authorization": f"Key {FAL_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model_image": to_base64(user_img),
                    "garment_image": to_base64(product_img),
                    "category": "tops"
                }
            )

        if response.status_code != 200:
            return JSONResponse({"error": response.text}, status_code=502)

        data = response.json()
        result_url = data["images"][0]["url"]

        # Download result image
        async with httpx.AsyncClient() as client:
            img_response = await client.get(result_url)

        result_path = os.path.join(UPLOAD_DIR, f"{job_id}_result.png")
        with open(result_path, "wb") as f:
            f.write(img_response.content)

        return {
            "status": "success",
            "job_id": job_id,
            "result_image": f"/result/{job_id}"
        }

    except Exception as e:
        import traceback
        print("ERROR:", traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    path = os.path.join(UPLOAD_DIR, f"{job_id}_result.png")
    if not os.path.exists(path):
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(path)