import base64
import os
import json
import time
import requests

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

from openai import OpenAI
from dotenv import load_dotenv

from story import STORY

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------
# Setup
# -----------------------------
load_dotenv()

app = FastAPI()

@app.middleware("http")
async def log_requests(request, call_next):
    print("🌐 REQUEST:", request.url)
    return await call_next(request)

OUTPUT_DIR = "generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/generated", StaticFiles(directory=OUTPUT_DIR), name="generated")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LEONARDO_API_KEY = os.getenv("LEONARDO_API_KEY")

if not LEONARDO_API_KEY:
    print("❌ ERROR: LEONARDO_API_KEY missing")

LEONARDO_HEADERS = {
    "accept": "application/json",
    "authorization": f"Bearer {LEONARDO_API_KEY}",
    "content-type": "application/json"
}

# -----------------------------
# Prompt
# -----------------------------
BASE_CHARACTER = """
Same 10-year-old boy Yosei,
short black hair, soft eyes, slightly shy expression,
wearing a light blue t-shirt, brown shorts, school bag,
consistent character, children's book illustration,
watercolor, pastel colors
"""

# =========================================================
# ✅ OPENAI STREAM
# =========================================================
@app.get("/stream-story")
def stream_story():

    def generate():
        print("🚀 OPENAI STREAM STARTED")

        for i, item in enumerate(STORY):
            yield ": heartbeat\n\n"

            file_path = f"{OUTPUT_DIR}/page_{i}.png"

            if not os.path.exists(file_path):
                print(f"🔥 OpenAI generating page {i}")

                prompt = f"{BASE_CHARACTER} {item['scene']}"

                try:
                    result = client.images.generate(
                        model="gpt-image-1",
                        prompt=prompt,
                        size="1024x1024"
                    )

                    img_base64 = result.data[0].b64_json
                    img_bytes = base64.b64decode(img_base64)

                    with open(file_path, "wb") as f:
                        f.write(img_bytes)

                    print(f"✅ OpenAI saved: {file_path}")

                except Exception as e:
                    print("❌ OpenAI ERROR:", e)

            data = {
                "image": f"http://127.0.0.1:8000/generated/page_{i}.png",
                "text": item["text"],
                "index": i,
                "source": "openai"
            }

            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.1)

        yield "event: end\ndata: done\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# =========================================================
# ✅ LEONARDO GENERATION
# =========================================================
def generate_image_leonardo(scene, index):

    file_path = f"{OUTPUT_DIR}/leo_page_{index}.png"

    if os.path.exists(file_path):
        print(f"✅ Leonardo cached: {file_path}")
        return file_path

    prompt = f"{BASE_CHARACTER} {scene}"

    payload = {
        "prompt": prompt,
        "modelId": "b2614463-296c-462a-9586-aafdb8f00e36",
        "width": 1024,
        "height": 1024,
        "num_images": 1
    }

    print("🔥 Leonardo generating:", index)

    response = requests.post(
        "https://cloud.leonardo.ai/api/rest/v1/generations",
        json=payload,
        headers=LEONARDO_HEADERS
    )

    print("📡 Leonardo response:", response.text)

    data = response.json()

    try:
        generation_id = data["sdGenerationJob"]["generationId"]
    except Exception as e:
        raise Exception(f"Leonardo failed: {data}")

    image_url = None

    for attempt in range(20):
        time.sleep(2)

        status = requests.get(
            f"https://cloud.leonardo.ai/api/rest/v1/generations/{generation_id}",
            headers=LEONARDO_HEADERS
        ).json()

        print(f"⏳ Leonardo polling {attempt}:", status)

        try:
            images = status["generations_by_pk"]["generated_images"]
            if images:
                image_url = images[0]["url"]
                break
        except Exception as e:
            print("⚠️ Polling error:", e)

    if not image_url:
        raise Exception("❌ Leonardo generation timeout")

    img_bytes = requests.get(image_url).content

    with open(file_path, "wb") as f:
        f.write(img_bytes)

    print(f"✅ Leonardo saved: {file_path}")

    return file_path


# =========================================================
# ✅ LEONARDO STREAM
# =========================================================
@app.get("/stream-story-leonardo")
def stream_story_leonardo():

    def generate():
        print("🚀 LEONARDO STREAM STARTED")

        for i, item in enumerate(STORY):
            yield ": heartbeat\n\n"

            try:
                generate_image_leonardo(item["scene"], i)

                data = {
                    "image": f"http://127.0.0.1:8000/generated/leo_page_{i}.png",
                    "text": item["text"],
                    "index": i,
                    "source": "leonardo"
                }

                yield f"data: {json.dumps(data)}\n\n"

            except Exception as e:
                print("❌ Leonardo ERROR:", e)

            time.sleep(0.1)

        yield "event: end\ndata: done\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# =========================================================
# ✅ EXPORT PDF
# =========================================================
@app.get("/export-pdf")
def export_pdf():

    pdf_path = f"{OUTPUT_DIR}/storybook.pdf"

    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()

    story_flow = []

    for i, item in enumerate(STORY):
        img_path = os.path.join(BASE_DIR, OUTPUT_DIR, f"page_{i}.png")

        if not os.path.exists(img_path):
            raise Exception(f"Image not ready: {img_path}")

        story_flow.append(RLImage(img_path, width=400, height=400))
        story_flow.append(Spacer(1, 20))
        story_flow.append(Paragraph(item["text"], styles["Normal"]))
        story_flow.append(Spacer(1, 40))

    doc.build(story_flow)

    return {"pdf": pdf_path}