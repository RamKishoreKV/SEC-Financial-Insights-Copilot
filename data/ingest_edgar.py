"""
Minimal EDGAR ingestion for this demo.
- Downloads a small set of public 10-K/10-Q filings.
- Extracts simple sections (Item 1A/7) heuristically.
- Writes documents + embeddings into local Chroma persist dir (same as retrieval service).

Note: Requires internet access. If offline, skip running; the app will keep using the seeded sample docs.
"""

import argparse
import os
import re
import time
from pathlib import Path
from io import BytesIO
import zipfile
import requests
from chromadb import Client
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from pathlib import Path

DEFAULT_DOCS = [
    {
        "id": "aapl-2023-10k",
        "urls": [
            # SEC bulk ZIP (plain text inside)
            "https://www.sec.gov/Archives/edgar/data/320193/000032019323000105/0000320193-23-000105.txt",
        ],
        "company": "Apple Inc",
        "year": 2023,
        "source": "AAPL 10-K 2023",
    },
    {
        "id": "msft-2023-10k",
        "urls": [
            "https://www.sec.gov/Archives/edgar/data/789019/000156459023009701/0001564590-23-009701.txt",
        ],
        "company": "Microsoft Corp",
        "year": 2023,
        "source": "MSFT 10-K 2023",
    },
]

# Fallback snippets if remote fetch fails (to stay functional offline)
FALLBACK_TEXT = {
    "aapl-2023-10k": """
    Item 7. Management’s Discussion and Analysis.
    Net sales increased to $383.3 billion in 2023, up 2% year-over-year,
    driven by Services and iPhone. Gross margin expanded to 44.1%.
    Risk factors include supply chain disruptions and regulatory scrutiny.
    """,
    "msft-2023-10k": """
    Item 7. Management’s Discussion and Analysis.
    Revenue reached $211.9 billion in 2023, a 7% increase, driven by Cloud (Azure).
    Operating income grew 6% to $88.5 billion. Risks include competitive cloud pricing,
    cybersecurity threats, and evolving AI regulation.
    """,
}


UA = os.environ.get("SEC_USER_AGENT", "kv.ramkishore0@gmail.com research-bot/0.1")


SAMPLES_DIR = Path(__file__).parent / "edgar_samples"


def load_local_sample(doc_id: str) -> str | None:
    path = SAMPLES_DIR / f"{doc_id.split('-')[0]}_2023_item1a.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def fetch_any(urls: list[str]) -> str:
    headers = {"User-Agent": UA}
    last_err = None
    for url in urls:
        try:
            if url.endswith(".zip"):
                resp = requests.get(url, headers=headers, timeout=120)
                resp.raise_for_status()
                with zipfile.ZipFile(BytesIO(resp.content)) as zf:
                    txt_names = [n for n in zf.namelist() if n.lower().endswith(".txt")]
                    target = txt_names[0] if txt_names else zf.namelist()[0]
                    with zf.open(target) as f:
                        return f.read().decode("utf-8", errors="ignore")
            resp = requests.get(url, headers=headers, timeout=60)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            last_err = exc
            continue
    if last_err:
        raise last_err
    raise RuntimeError("No URLs provided")


def extract_sections(html: str) -> list[dict]:
    # Heuristic: pull ITEM 1A (risk) and ITEM 7 (MD&A) text blocks
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    sections = []
    patterns = [
        ("Item 1A", r"item\s*1a\.\s*(.+?)item\s*1b", 2000),
        ("Item 7", r"item\s*7\.\s*(.+?)item\s*7a", 2000),
    ]
    for label, pattern, max_len in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            chunk = m.group(1)[: max_len]
            sections.append((label, chunk))
    return [{"section": s[0], "content": s[1]} for s in sections]


def ingest(docs, persist_dir: Path):
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = Client(ChromaSettings(persist_directory=str(persist_dir), anonymized_telemetry=False))
    collection = client.get_or_create_collection("sec-filings", metadata={"hnsw:space": "cosine"})
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    added = 0
    for doc in docs:
        html = None
        # Try local sample first
        local = load_local_sample(doc["id"])
        if local:
            html = local
        else:
            try:
                print(f"Downloading {doc['id']}")
                html = fetch_any(doc["urls"])
            except Exception as exc:
                print(f"  Fetch failed ({exc}); using fallback text for {doc['id']}")

        sections = extract_sections(html) if html else []
        if not sections:
            fallback = FALLBACK_TEXT.get(doc["id"])
            if fallback:
                sections = [{"section": "MD&A", "content": fallback}]
        if not sections:
            print(f"  No sections extracted; skipping {doc['id']}")
            continue
        for idx, sec in enumerate(sections):
            doc_id = f"{doc['id']}-{sec['section'].lower().replace(' ', '-')}-{idx}"
            text = f"{sec['section']} | {sec['content']}"
            embedding = embedder.encode([text], convert_to_numpy=True).tolist()[0]
            collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[{"company": doc["company"], "year": doc["year"], "source": doc["source"]}],
                embeddings=[embedding],
            )
            added += 1
    print(f"Ingested {added} chunks into {persist_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist_dir", default="./data/chroma", help="Chroma persist directory (match retrieval)")
    parser.add_argument("--max_docs", type=int, default=len(DEFAULT_DOCS))
    args = parser.parse_args()

    docs = DEFAULT_DOCS[: args.max_docs]
    start = time.time()
    ingest(docs, Path(args.persist_dir))
    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()

