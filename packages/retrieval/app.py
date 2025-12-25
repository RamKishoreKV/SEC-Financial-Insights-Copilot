import json
from pathlib import Path
import os
from typing import List
from uuid import uuid4
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from common import Settings, QARequest, RetrievalDocument, RetrievalResult, get_logger
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import requests
from io import BytesIO
import re


settings = Settings()
logger = get_logger("retrieval")
app = FastAPI(title="Retrieval Service", version="0.1.0")
SEC_UA = os.getenv("SEC_USER_AGENT", "kv.ramkishore0@gmail.com sec-copilot/0.1")


def resolve_data_path() -> Path:
    candidates = [
        Path("/app/data/sec_samples.json"),
        Path.cwd() / "data" / "sec_samples.json",
    ]
    # Safely add parent traversal if available
    try:
        parent_data = Path(__file__).resolve().parents[1] / "data" / "sec_samples.json"
        candidates.append(parent_data)
    except Exception:
        pass
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


DATA_PATH = resolve_data_path()


def load_corpus() -> List[dict]:
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load corpus: %s", exc)
        return []


CORPUS = load_corpus()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client(
    ChromaSettings(
        persist_directory=settings.chroma_persist_dir,
        anonymized_telemetry=False,
    )
)
collection = chroma_client.get_or_create_collection(
    name="sec-filings",
    metadata={"hnsw:space": "cosine"},
)


def seed_collection():
    if not CORPUS:
        return
    if collection.count() > 0:
        return
    documents = [doc["content"] for doc in CORPUS]
    ids = [doc["id"] for doc in CORPUS]
    metadatas = [
        {"company": doc.get("company"), "year": doc.get("year"), "source": doc["source"], "id": doc["id"]}
        for doc in CORPUS
    ]
    embeddings = embedder.encode(documents, convert_to_numpy=True).tolist()
    collection.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)
    logger.info("Seeded %d documents into Chroma", len(ids))


try:
    seed_collection()
except Exception as exc:
    logger.error("Seeding failed, will fall back to lexical: %s", exc)


def lexical_score(question: str, text: str) -> float:
    q_tokens = set(question.lower().split())
    t_tokens = set(text.lower().split())
    overlap = q_tokens.intersection(t_tokens)
    return len(overlap) / (len(q_tokens) + 1e-6)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def extract_text(file: UploadFile) -> str:
    data = file.file.read()
    return extract_text_bytes(data, file.filename or "upload")


def extract_text_bytes(data: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    text = data.decode("utf-8", errors="ignore")
    # If HTML, strip tags
    if filename.lower().endswith((".htm", ".html")) or "<html" in text.lower():
        text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


@app.get("/health")
async def health():
    return {"status": "ok", "documents": len(CORPUS)}


@app.post("/retrieve", response_model=RetrievalResult)
async def retrieve(req: QARequest):
    # If no seeded corpus and index empty, error
    if not CORPUS and collection.count() == 0:
        raise HTTPException(status_code=500, detail="Corpus not loaded")

    # Vector search path
    try:
        where = {}
        if req.company:
            where["company"] = req.company
        if req.year:
            where["year"] = req.year
        query_embedding = embedder.encode([req.question], convert_to_numpy=True).tolist()[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            where=where if where else None,
        )
        documents = []
        for doc, meta, score in zip(
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0],
        ):
            documents.append(
                RetrievalDocument(
                    id=meta.get("id") or meta.get("source", "doc"),
                    content=doc,
                    source=meta.get("source", "unknown"),
                    score=float(score),
                )
            )
        if documents:
            return RetrievalResult(documents=documents)
    except Exception as exc:
        logger.error("Vector search failed, falling back to lexical: %s", exc)

    # Lexical fallback
    filtered = [
        doc
        for doc in CORPUS
        if (not req.company or doc.get("company") == req.company)
        and (not req.year or doc.get("year") == req.year)
    ]
    if not filtered:
        filtered = CORPUS

    scored = [
        (
            doc,
            lexical_score(req.question, doc["content"]) + lexical_score(req.question, doc["source"]),
        )
        for doc in filtered
    ]
    scored = sorted(scored, key=lambda x: x[1], reverse=True)[:3]

    documents = [
        RetrievalDocument(
            id=doc["id"],
            content=doc["content"],
            source=doc["source"],
            score=score,
        )
        for doc, score in scored
    ]
    return RetrievalResult(documents=documents)


@app.post("/ingest")
async def ingest(
    file: UploadFile | None = File(None),
    url: List[str] | None = Form(None),
    company: str = Form("upload"),
    year: int = Form(0),
):
    """
    Upload a document (PDF or text) OR provide a URL to fetch, then add chunks to the vector index.
    """
    try:
        if not file and not url:
            raise HTTPException(status_code=400, detail="Provide a file or at least one url.")

        text = ""
        source_name = ""
        if file:
            text = extract_text(file)
            source_name = file.filename or "upload"
            if not text.strip():
                raise ValueError("Uploaded file is empty or not text-decodable.")
            chunks = chunk_text(text)
            if not chunks:
                raise ValueError("No text chunks extracted from upload.")
            ids = []
            docs = []
            metas = []
            for chunk in chunks:
                doc_id = f"upload-{uuid4().hex}"
                ids.append(doc_id)
                docs.append(chunk)
                metas.append(
                    {
                        "company": company,
                        "year": year,
                        "source": source_name,
                    }
                )
            embeds = embedder.encode(docs, convert_to_numpy=True).tolist()
            collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)
            return {"status": "ok", "chunks": len(chunks)}

        # Handle multiple URLs
        if not url:
            raise HTTPException(status_code=400, detail="No url provided.")
        headers = {"User-Agent": SEC_UA}
        all_chunks = 0
        for u in url:
            resp = requests.get(u, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.content
            source_name = Path(u).name or "download"
            text = extract_text_bytes(data, source_name)
            if not text.strip():
                continue
            chunks = chunk_text(text)
            if not chunks:
                continue
            ids = []
            docs = []
            metas = []
            for chunk in chunks:
                doc_id = f"url-{uuid4().hex}"
                ids.append(doc_id)
                docs.append(chunk)
                metas.append(
                    {
                        "company": company,
                        "year": year,
                        "source": source_name,
                    }
                )
            embeds = embedder.encode(docs, convert_to_numpy=True).tolist()
            collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)
            all_chunks += len(chunks)
        if all_chunks == 0:
            raise HTTPException(status_code=400, detail="No text extracted from provided URLs.")
        return {"status": "ok", "chunks": all_chunks}
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc))


