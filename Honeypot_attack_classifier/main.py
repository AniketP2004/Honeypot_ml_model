from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src import models
from src.features import build_feature_row

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

from pydantic import BaseModel

class RawPacket(BaseModel):
    src_ip: str
    dst_port: int
    protocol: str
    timestamp: str
    ttl: int | None = None
    window_size: int | None = None
    tcp_flag: str | None = None
    src_port: int
    http_method: str = ""
    http_path: str = ""
    user_agent: str = ""
    ssh_client: str = ""
    payload_size: int | None = None
    request_rate_override: int | None = None

@app.on_event("startup")
def startup():
    models.load_all()

@app.post("/classify")
def classify(pkt: RawPacket, model: str="gradient_boosting"):
    if model not in ("random_forest", "xgboost", "gradient_boosting"):
        raise HTTPException(400, "invalid model")

    df = build_feature_row(pkt.dict())
    df = df[models.REG["preprocessor"].feature_names_in_]
    X = models.REG["preprocessor"].transform(df)

    clf = models.REG[model]
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0].max()
    label = models.REG["le_multi"].inverse_transform([pred])[0]

    bclf = models.REG["binary"]
    bpred = bclf.predict(X)[0]
    bproba = bclf.predict_proba(X)[0].max()
    blabel = models.REG["le_binary"].inverse_transform([bpred])[0]

    return {
        "model_used": model,
        "attack_type": label,
        "attack_type_confidence": float(proba),
        "binary_verdict": blabel,
        "binary_confidence": float(bproba)

    }

@app.get("/models")
def list_models():
    return{
        k: {"loaded": k in models.REG, "classes": models.REG[k].classes_.tolist()}
        for k in ("random_forest", "xgboost", "gradient_boosting", "binary")
        if k in models.REG
    }

@app.get("/health")
def health():
    required = ["random_forest", "xgboost", "gradient_boosting", "binary", "preprocessor", "le_multi", "le_binary"]
    return{"status": "ok" if all(r in models.REG for r in required) else "degraded",
           "loaded": {r:r in models.REG for r in required}}

@app.post("/reload-models")
def reload_models():
    models.reload()
    return {"status": "reloaded"}
