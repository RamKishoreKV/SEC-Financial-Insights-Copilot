import uuid
from fastapi import FastAPI, Response
from pydantic import BaseModel
from common import EvalResult, get_logger


logger = get_logger("evaluator")
app = FastAPI(title="Evaluator Service", version="0.1.0")


class EvalRequest(BaseModel):
    question: str
    answer: str
    documents: list


def run_heuristics(req: EvalRequest) -> EvalResult:
    reasons = []
    risk = "low"

    has_citation = any(doc.get("source") in req.answer for doc in req.documents)
    if not has_citation:
        reasons.append("Answer lacks explicit citation string.")
        risk = "medium"

    # Very light numeric consistency check: if question asks revenue, ensure a number is present
    if "revenue" in req.question.lower() and not any(char.isdigit() for char in req.answer):
        reasons.append("Revenue question answered without numeric figure.")
        risk = "high"

    passed = len(reasons) == 0
    if passed:
        reasons.append("Heuristics passed.")

    return EvalResult(passed=passed, reasons=reasons, risk_level=risk)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvalResult)
async def evaluate(req: EvalRequest, response: Response):
    trace_id = str(uuid.uuid4())
    result = run_heuristics(req)
    if response is not None:
        response.headers["x-trace-id"] = trace_id
    return result

