from __future__ import annotations

from typing import TypedDict, List
import httpx
from fastapi import FastAPI, HTTPException
from langgraph.graph import StateGraph, START, END
from common import (
    Settings,
    QARequest,
    QAResponse,
    RetrievalResult,
    RetrievalDocument,
    EvalResult,
    LLMRouter,
    get_logger,
    Timings,
)
import time


class OrchestratorState(TypedDict, total=False):
    question: str
    company: str | None
    year: int | None
    documents: List[dict]
    answer: str
    eval: dict
    provider: str
    trace_id: str
    retrieve_ms: float
    generate_ms: float
    eval_ms: float


settings = Settings()
logger = get_logger("orchestrator")
llm_router = LLMRouter()
app = FastAPI(title="Orchestrator", version="0.1.0")


async def retrieve_node(state: OrchestratorState) -> OrchestratorState:
    start = time.perf_counter()
    payload = {
        "question": state["question"],
        "company": state.get("company"),
        "year": state.get("year"),
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{settings.retrieval_url}/retrieve", json=payload)
    resp.raise_for_status()
    result = RetrievalResult(**resp.json())
    elapsed = (time.perf_counter() - start) * 1000.0
    return {**state, "documents": [doc.dict() for doc in result.documents], "retrieve_ms": elapsed}


async def generate_node(state: OrchestratorState) -> OrchestratorState:
    start = time.perf_counter()
    docs = state.get("documents", [])
    context = "\n\n".join([f"{d['source']}: {d['content']}" for d in docs])
    prompt = (
        "You are a financial analyst. Answer with citations to the provided SEC context.\n"
        f"Question: {state['question']}\n"
        f"Context:\n{context}\n"
        "Respond with an evidence-backed answer."
    )
    answer = await llm_router.generate(prompt)
    elapsed = (time.perf_counter() - start) * 1000.0
    return {**state, "answer": answer, "provider": settings.default_provider, "generate_ms": elapsed}


async def eval_node(state: OrchestratorState) -> OrchestratorState:
    start = time.perf_counter()
    payload = {
        "answer": state["answer"],
        "question": state["question"],
        "documents": state.get("documents", []),
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{settings.evaluator_url}/evaluate", json=payload)
    resp.raise_for_status()
    eval_result = EvalResult(**resp.json())
    elapsed = (time.perf_counter() - start) * 1000.0
    return {
        **state,
        "eval": eval_result.dict(),
        "trace_id": resp.headers.get("x-trace-id", ""),
        "eval_ms": elapsed,
    }


def build_graph():
    graph = StateGraph(OrchestratorState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("evaluate", eval_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_edge("evaluate", END)
    return graph.compile()


graph = build_graph()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/orchestrate", response_model=QAResponse)
async def orchestrate(req: QARequest):
    try:
        t0 = time.perf_counter()
        initial_state: OrchestratorState = {
            "question": req.question,
            "company": req.company,
            "year": req.year,
        }
        result: OrchestratorState = await graph.ainvoke(initial_state)
        t_total = (time.perf_counter() - t0) * 1000.0
        citations = [
            RetrievalDocument(**doc) for doc in result.get("documents", [])
        ]
        timings = Timings(
            retrieve_ms=result.get("retrieve_ms", 0.0),
            generate_ms=result.get("generate_ms", 0.0),
            eval_ms=result.get("eval_ms", 0.0),
            total_ms=t_total,
        )
        return QAResponse(
            answer=result["answer"],
            citations=citations,
            eval=EvalResult(**result["eval"]),
            provider=result.get("provider", settings.default_provider),
            trace_id=result.get("trace_id", ""),
            timings=timings,
        )
    except Exception as exc:
        logger.exception("Orchestration failed")
        raise HTTPException(status_code=500, detail=str(exc))

