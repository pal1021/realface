# backend/main.py
import os, tempfile, cv2, numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import InferenceApi

app = FastAPI()
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
app.add_middleware(CORSMiddleware, allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"], allow_methods=["*"], allow_headers=["*"])

HF_TOKEN = os.getenv("HF_TOKEN")
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "prithivMLmods/deepfake-detector-model-v1")  # default example

if HF_TOKEN:
    image_infer = InferenceApi(repo_id=HF_IMAGE_MODEL, token=HF_TOKEN)
else:
    image_infer = None

def parse_hf_resp(resp):
    # try common shapes
    try:
        if isinstance(resp, list):
            # look for 'fake' label or return top score
            for item in resp:
                lab = str(item.get("label","")).lower()
                if "fake" in lab or "ai" in lab or "synthetic" in lab:
                    return float(item.get("score", 0.0))
            return float(resp[0].get("score", 0.0))
        if isinstance(resp, dict):
            if "score" in resp:
                return float(resp.get("score", 0.0))
    except Exception:
        pass
    return 0.5

async def detect_image(path):
    if not image_infer:
        raise RuntimeError("Hugging Face not configured (HF_TOKEN).")
    with open(path, "rb") as f:
        img = f.read()
    resp = image_infer(inputs=img)
    score = parse_hf_resp(resp)
    return float(score), {"hf_raw": resp}

@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "")[-1] or ".jpg")
    tmp.write(await file.read())
    tmp.close()
    try:
        score, detail = await detect_image(tmp.name)
        return JSONResponse({"type":"image","score":score,"detail":detail})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try: os.remove(tmp.name)
        except: pass

def extract_frames(path, max_frames=8):
    frames=[]
    cap=cv2.VideoCapture(path)
    if not cap.isOpened(): return frames
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step=max(1, total//max_frames) if total>0 else 1
    i=0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if i % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(frames) >= max_frames: break
        i += 1
    cap.release()
    return frames

@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(await file.read())
    tmp.close()
    try:
        frames = extract_frames(tmp.name, max_frames=6)
        results=[]
        for idx, fr in enumerate(frames):
            # save frame and call image inference
            ftmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(ftmp.name, cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            score, detail = await detect_image(ftmp.name)
            results.append({"frame": idx, "score": score, "detail": detail})
            try: os.remove(ftmp.name)
            except: pass
        mean_score = float(np.mean([r["score"] for r in results])) if results else 0.0
        return JSONResponse({"type":"video","frames_analyzed":len(results),"mean_score":mean_score,"frame_reports":results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try: os.remove(tmp.name)
        except: pass

@app.post("/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    # For beginners we return a message; advanced: transcribe + text model
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "")[-1] or ".wav")
    tmp.write(await file.read())
    tmp.close()
    try:
        return JSONResponse({"type":"audio","score":None,"detail":"Audio support: set HF_AUDIO_MODEL or transcribe then classify. See docs."})
    finally:
        try: os.remove(tmp.name)
        except: pass

@app.get("/")
def root():
    return {"ok": True}

