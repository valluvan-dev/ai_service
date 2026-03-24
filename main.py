from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import shutil, os, uuid
from PIL import Image
import numpy as np
from rembg import remove
import cv2
import mediapipe as mp

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------
# 🔥 MEDIAPIPE
# -------------------------------
mp_pose = mp.solutions.pose

def get_shoulders(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("Image not loaded")

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
# 🔥 SEGMENT
# -------------------------------
def segment_person(image_path):
    return remove(Image.open(image_path).convert("RGBA"))


# -------------------------------
# 🔥 TRYON API
# -------------------------------
@app.post("/tryon")
async def tryon(user_image: UploadFile = File(...), product_image: UploadFile = File(...)):

    job_id = str(uuid.uuid4())

    user_path = os.path.join(UPLOAD_DIR, f"{job_id}_user.png")
    product_path = os.path.join(UPLOAD_DIR, f"{job_id}_product.png")
    result_path = os.path.join(UPLOAD_DIR, f"{job_id}_result.png")

    # save images
    with open(user_path, "wb") as f:
        shutil.copyfileobj(user_image.file, f)

    with open(product_path, "wb") as f:
        shutil.copyfileobj(product_image.file, f)

    try:
        print("🔥 Step 1: Segmentation")
        person_img = segment_person(user_path)
        person_np = np.array(person_img)
        person_rgb = person_np[:, :, :3]

        print("🔥 Step 2: Pose Detection")
        shoulders = get_shoulders(user_path)
        if shoulders is None:
            raise Exception("Pose not detected")

        left, right = shoulders

        print("🔥 Step 3: Cloth Resize")
        shoulder_width = abs(right[0] - left[0])
        cloth_width = int(shoulder_width * 1.5)
        cloth_height = int(cloth_width * 1.3)

        cloth = Image.open(product_path).convert("RGBA")
        cloth_np = np.array(cloth.resize((cloth_width, cloth_height)))

        print("🔥 Step 4: Warping")
        src = np.float32([[0,0],[cloth_width,0],[0,cloth_height],[cloth_width,cloth_height]])

        top_y = int(min(left[1], right[1]) - cloth_height * 0.3)

        dst = np.float32([
            [left[0], top_y],
            [right[0], top_y],
            [left[0], top_y + cloth_height],
            [right[0], top_y + cloth_height]
        ])

        matrix = cv2.getPerspectiveTransform(src, dst)

        warped = cv2.warpPerspective(
            cloth_np,
            matrix,
            (person_rgb.shape[1], person_rgb.shape[0])
        )

        print("🔥 Step 5: Blending")
        alpha = warped[:, :, 3] / 255.0

        for c in range(3):
            person_rgb[:, :, c] = (
                alpha * warped[:, :, c] +
                (1 - alpha) * person_rgb[:, :, c]
            )

        Image.fromarray(person_rgb.astype(np.uint8)).save(result_path)

    except Exception as e:
        print("❌ ERROR:", str(e))
        return JSONResponse({"error": str(e)}, status_code=500)

    return {
        "status": "success",
        "job_id": job_id,
        "result_image": f"/result/{job_id}"
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
