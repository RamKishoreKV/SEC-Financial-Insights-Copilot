from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import httpx
from common import Settings, QARequest, QAResponse, get_logger


settings = Settings()
logger = get_logger("api-gateway")
app = FastAPI(title="SEC Filings Gateway", version="0.1.0")

origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/qa", response_model=QAResponse)
async def qa(request: QARequest):
    """
    Entry point for financial Q&A. Forwards to orchestrator.
    """
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{settings.orchestrator_url}/orchestrate", json=request.dict()
            )
        resp.raise_for_status()
        return QAResponse(**resp.json())
    except Exception as exc:
        logger.exception("QA request failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest")
async def ingest(
    file: UploadFile | None = File(None),
    url: list[str] | None = Form(None),
    company: str | None = Form(None),
    year: int | None = Form(None),
):
    """
    Proxy file upload to retrieval service.
    """
    if file is None and not url:
        raise HTTPException(status_code=400, detail="Provide a file or at least one url.")
    try:
        multipart = []
        multipart.append(("company", (None, company or "upload")))
        multipart.append(("year", (None, str(year or 0))))
        if file:
            content = await file.read()
            multipart.append(
                (
                    "file",
                    (
                        file.filename or "upload.bin",
                        content,
                        file.content_type or "application/octet-stream",
                    ),
                )
            )
        if url:
            for u in url:
                multipart.append(("url", (None, u)))
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{settings.retrieval_url}/ingest",
                files=multipart,
            )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.exception("Ingest request failed")
        raise HTTPException(status_code=500, detail=str(exc))



