from pydantic import BaseModel, Field
from typing import List, Optional


class Timings(BaseModel):
    retrieve_ms: float
    generate_ms: float
    eval_ms: float
    total_ms: float


class QARequest(BaseModel):
    question: str = Field(..., description="User question about SEC filings")
    company: Optional[str] = Field(None, description="Optional company filter/ticker")
    year: Optional[int] = Field(None, description="Optional filing year filter")


class RetrievalDocument(BaseModel):
    id: str
    content: str
    source: str
    score: float


class RetrievalResult(BaseModel):
    documents: List[RetrievalDocument]


class EvalResult(BaseModel):
    passed: bool
    reasons: List[str]
    risk_level: str = "low"


class QAResponse(BaseModel):
    answer: str
    citations: List[RetrievalDocument]
    eval: EvalResult
    provider: str
    trace_id: str
    timings: Optional[Timings] = None

