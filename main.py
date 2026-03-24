from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import shutil, os, uuid, httpx

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

FASHN_API_URL = "https://inspectional-pseudolinguistically-halley.ngrok-free.dev/tryon"


# -------------------------------
# TRYON API
# -------------------------------
@app.post("/tryon")
async def tryon(user_image: UploadFile = File(...), product_image: UploadFile = File(...)):
    job_id = str(uuid.uuid4())

    user_path = os.path.join(UPLOAD_DIR, f"{job_id}_user.png")
    product_path = os.path.join(UPLOAD_DIR, f"{job_id}_product.png")

    with open(user_path, "wb") as f:
        shutil.copyfileobj(user_image.file, f)

    with open(product_path, "wb") as f:
        shutil.copyfileobj(product_image.file, f)

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            with open(user_path, "rb") as u, open(product_path, "rb") as p:
                response = await client.post(
                    FASHN_API_URL,
                    files={
                        "user_image": ("user.png", u, "image/png"),
                        "product_image": ("product.png", p, "image/png"),
                    }
                )

        if response.status_code != 200:
            return JSONResponse({"error": f"FASHN API error: {response.text}"}, status_code=502)

        data = response.json()
        base64_image = data.get("result_image") or data.get("image") or data.get("base64")

        if not base64_image:
            return JSONResponse({"error": "No image in FASHN API response"}, status_code=502)

    except Exception as e:
        print("ERROR:", str(e))
        return JSONResponse({"error": str(e)}, status_code=500)

    return {
        "status": "success",
        "job_id": job_id,
        "result_image": base64_image,
    }


# -------------------------------
# RESULT
# -------------------------------
@app.get("/result/{job_id}")
async def get_result(job_id: str):
    path = os.path.join(UPLOAD_DIR, f"{job_id}_result.png")
    if not os.path.exists(path):
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(path)
