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
import asyncio


from story import STORY

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------
# Setup
# -----------------------------
load_dotenv()

app = FastAPI()

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

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LEONARDO_API_KEY = os.getenv("LEONARDO_API_KEY")

LEONARDO_HEADERS = {
    "accept": "application/json",
    "authorization": f"Bearer {LEONARDO_API_KEY}",
    "content-type": "application/json"
}

# -----------------------------
# ✅ COMMON PROMPT BUILDER
# -----------------------------
def build_prompt(page):
    return f"""
    {STORY['style']['art_style']},
    {STORY['main_character']['description']},
    {STORY['setting']},
    {page['image_prompt']},
    consistent character, same face, same outfit
    """

# =========================================================
# ✅ OPENAI STREAM
# =========================================================
@app.get("/stream-story")
def stream_story():

    def generate():
        print("🚀 OPENAI STREAM STARTED")

        for i, page in enumerate(STORY["pages"]):
            yield ": heartbeat\n\n"

            file_path = f"{OUTPUT_DIR}/page_{i}.png"

            try:
                if not os.path.exists(file_path):
                    print(f"🔥 OpenAI generating page {i}")

                    prompt = build_prompt(page)
                    try:

                        result = client.images.generate(
                            model="gpt-image-1",
                            prompt=prompt,
                            size="1024x1024"
                        )
                    except Exception as e:
                        print("timeout or error")
                    
                    img_base64 = result.data[0].b64_json
                    img_bytes = base64.b64decode(img_base64)

                    with open(file_path, "wb") as f:
                        f.write(img_bytes)

                    print(f"✅ OpenAI saved: {file_path}")

                else:
                    print(f"✅ OpenAI cached: {file_path}")

                data = {
                    "image": f"http://127.0.0.1:8000/generated/page_{i}.png",
                    "text": page["text"],
                    "index": i,
                    "source": "openai"
                }

            except Exception as e:
                print(f"❌ OpenAI ERROR page {i}:", e)

                # 👇 IMPORTANT: send fallback so UI doesn't hang
                data = {
                    "image": None,
                    "text": f"Error generating image (page {i})",
                    "index": i,
                    "source": "openai"
                }

            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.2)        

        yield "event: end\ndata: done\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# =========================================================
# ✅ LEONARDO IMAGE GENERATION
# =========================================================
def generate_image_leonardo(page, index):

    file_path = f"{OUTPUT_DIR}/leo_page_{index}.png"

    if os.path.exists(file_path):
        print(f"✅ Leonardo cached: {file_path}")
        return file_path

    prompt = build_prompt(page)

    payload = {
        "prompt": prompt,
        "modelId": "b2614463-296c-462a-9586-aafdb8f00e36",
        "width": 1024,
        "height": 1024,
        "num_images": 1
    }

    print(f"🔥 Leonardo generating page {index}")

    response = requests.post(
        "https://cloud.leonardo.ai/api/rest/v1/generations",
        json=payload,
        headers=LEONARDO_HEADERS
    )

    data = response.json()

    try:
        generation_id = data["sdGenerationJob"]["generationId"]
    except:
        raise Exception(f"Leonardo failed: {data}")

    image_url = None

    for _ in range(20):
        time.sleep(2)

        status = requests.get(
            f"https://cloud.leonardo.ai/api/rest/v1/generations/{generation_id}",
            headers=LEONARDO_HEADERS
        ).json()

        try:
            images = status["generations_by_pk"]["generated_images"]
            if images:
                image_url = images[0]["url"]
                break
        except:
            pass

    if not image_url:
        raise Exception("❌ Leonardo timeout")

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

        for i, page in enumerate(STORY["pages"]):
            yield ": heartbeat\n\n"

            try:
                generate_image_leonardo(page, i)

                data = {
                    "image": f"http://127.0.0.1:8000/generated/leo_page_{i}.png",
                    "text": page["text"],
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

    for i, page in enumerate(STORY["pages"]):
        img_path = f"{OUTPUT_DIR}/page_{i}.png"

        if not os.path.exists(img_path):
            raise Exception(f"Image missing: {img_path}")

        story_flow.append(RLImage(img_path, width=400, height=400))
        story_flow.append(Spacer(1, 20))
        story_flow.append(Paragraph(page["text"], styles["Normal"]))
        story_flow.append(Spacer(1, 40))

    doc.build(story_flow)

    return {"pdf": pdf_path}