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

from story import STORY12

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
# Benchmark Settings
# -----------------------------
USE_CACHE = False   # 🔥 Set False for real speed testing
LOG_FILE = "performance_log.json"

# =========================================================
# Seed Strategy
# =========================================================
def get_seed(index):
    teacher_pages = [5, 11]
    return 222 if index in teacher_pages else 111

# =========================================================
# Prompt Builder
# =========================================================
def build_prompt(page):
    return f"""
{STORY12["global_prompt_rules"]}

Character:
{STORY12['main_character']['description']}

Teacher:
{STORY12.get('teacher_character', {}).get('description', '')}

Scene:
{page['image_prompt']}

Style:
{STORY12['style']['art_style']}

Composition:
simple, clean, focused on main subject, gentle storytelling mood
"""

# =========================================================
# Leonardo Negative Prompt
# =========================================================
LEONARDO_NEGATIVE = """
duplicate characters, extra children, crowded scenes,
classroom setting, group of kids, anime style, 3d render
"""

# =========================================================
# Logging Helper
# =========================================================
def log_performance(entry):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

# =========================================================
# OpenAI Stream
# =========================================================
@app.get("/stream-story")
def stream_story():

    def generate():
        total_start = time.time()

        for i, page in enumerate(STORY12["pages"]):
            yield ": heartbeat\n\n"

            file_path = f"{OUTPUT_DIR}/v3_page_{i}.png"
            start_time = time.time()

            try:
                if not USE_CACHE or not os.path.exists(file_path):
                    print(f"🔥 OpenAI generating page {i}")

                    prompt = build_prompt(page)

                    for attempt in range(2):
                        try:
                            result = client.images.generate(
                                model="gpt-image-1",
                                prompt=prompt,
                                size="1024x1024"
                            )
                            break
                        except Exception as e:
                            print("Retry OpenAI...", e)
                            time.sleep(1)

                    img_base64 = result.data[0].b64_json
                    img_bytes = base64.b64decode(img_base64)

                    with open(file_path, "wb") as f:
                        f.write(img_bytes)

                generation_time = round(time.time() - start_time, 2)

                # log
                log_performance({
                    "page": i,
                    "source": "openai",
                    "generation_time": generation_time,
                    "timestamp": time.time()
                })

                data = {
                    "image": f"http://127.0.0.1:8000/generated/v3_page_{i}.png",
                    "text": page["text"],
                    "index": i,
                    "source": "openai",
                    "generation_time": generation_time,
                    "server_timestamp": time.time()
                }

            except Exception as e:
                print(f"❌ OpenAI ERROR page {i}:", e)

                data = {
                    "image": None,
                    "text": f"Error generating image (page {i})",
                    "index": i,
                    "source": "openai"
                }

            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.2)

        print("✅ OpenAI TOTAL TIME:", round(time.time() - total_start, 2), "s")
        yield "event: end\ndata: done\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# =========================================================
# Leonardo Image Generation
# =========================================================
def generate_image_leonardo(page, index):

    file_path = f"{OUTPUT_DIR}/v3_leo_page_{index}.png"
    start_time = time.time()

    if USE_CACHE and os.path.exists(file_path):
        return file_path, 0

    prompt = build_prompt(page)

    payload = {
        "prompt": prompt,
        "negative_prompt": LEONARDO_NEGATIVE,
        "modelId": "b2614463-296c-462a-9586-aafdb8f00e36",
        "width": 1024,
        "height": 1024,
        "num_images": 1,
        "seed": get_seed(index),
        "guidance_scale": 6,
        "num_inference_steps": 28
    }

    print(f"🔥 Leonardo generating page {index}")

    response = requests.post(
        "https://cloud.leonardo.ai/api/rest/v1/generations",
        json=payload,
        headers=LEONARDO_HEADERS
    )

    data = response.json()
    generation_id = data["sdGenerationJob"]["generationId"]

    image_url = None

    for _ in range(25):
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
        raise Exception("Leonardo timeout")

    img_bytes = requests.get(image_url).content

    with open(file_path, "wb") as f:
        f.write(img_bytes)

    generation_time = round(time.time() - start_time, 2)

    return file_path, generation_time

# =========================================================
# Leonardo Stream
# =========================================================
@app.get("/stream-story-leonardo")
def stream_story_leonardo():

    def generate():
        total_start = time.time()

        for i, page in enumerate(STORY12["pages"]):
            yield ": heartbeat\n\n"

            try:
                file_path, gen_time = generate_image_leonardo(page, i)

                log_performance({
                    "page": i,
                    "source": "leonardo",
                    "generation_time": gen_time,
                    "timestamp": time.time()
                })

                data = {
                    "image": f"http://127.0.0.1:8000/generated/v3_leo_page_{i}.png",
                    "text": page["text"],
                    "index": i,
                    "source": "leonardo",
                    "generation_time": gen_time,
                    "server_timestamp": time.time()
                }

                yield f"data: {json.dumps(data)}\n\n"

            except Exception as e:
                print("❌ Leonardo ERROR:", e)

            time.sleep(0.1)

        print("✅ Leonardo TOTAL TIME:", round(time.time() - total_start, 2), "s")
        yield "event: end\ndata: done\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# =========================================================
# Export PDF
# =========================================================
@app.get("/export-pdf")
def export_pdf():

    pdf_path = f"{OUTPUT_DIR}/storybook.pdf"

    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()

    story_flow = []

    for i, page in enumerate(STORY12["pages"]):
        img_path = f"{OUTPUT_DIR}/v3_page_{i}.png"

        if not os.path.exists(img_path):
            raise Exception(f"Image missing: {img_path}")

        story_flow.append(RLImage(img_path, width=400, height=400))
        story_flow.append(Spacer(1, 20))
        story_flow.append(Paragraph(page["text"], styles["Normal"]))
        story_flow.append(Spacer(1, 40))

    doc.build(story_flow)

    return {"pdf": pdf_path}