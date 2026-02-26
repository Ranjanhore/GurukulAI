import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="GurukulAI Backend", version="debug-1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RespondReq(BaseModel):
    text: str

@app.get("/")
def root():
    return {"ok": True, "ts": int(time.time()), "routes": ["/health", "/respond"]}

@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}

@app.post("/respond")
def respond(req: RespondReq):
    return {"teacher_text": f"Echo: {req.text}", "ts": int(time.time())}
