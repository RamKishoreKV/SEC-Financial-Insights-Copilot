from __future__ import annotations

import httpx
from typing import Literal
from .config import get_settings
from .telemetry import get_logger


class LLMRouter:
    """
    Minimal provider router.
    - If MOCK_MODE=true, returns a canned response.
    - Otherwise routes to Ollama (local) or OpenAI (remote).
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger("llm")

    async def generate(self, prompt: str) -> str:
        if self.settings.mock_mode:
            return f"[MOCKED ANSWER] {prompt[:120]}"

        provider = self.settings.default_provider.lower()
        if provider == "ollama":
            return await self._call_ollama(prompt)
        if provider == "openai":
            return await self._call_openai(prompt)
        raise ValueError(f"Unknown provider: {provider}")

    async def _call_ollama(self, prompt: str) -> str:
        url = "http://host.docker.internal:11434/api/generate"
        payload = {"model": self.settings.ollama_model, "prompt": prompt, "stream": False}
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()

    async def _call_openai(self, prompt: str) -> str:
        # Lightweight call to OpenAI Chat Completion
        api_key = self.settings.openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        headers = {"Authorization": f"Bearer {api_key}"}
        json_payload = {
            "model": self.settings.openai_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(
            timeout=120, base_url="https://api.openai.com/v1"
        ) as client:
            resp = await client.post("/chat/completions", headers=headers, json=json_payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

