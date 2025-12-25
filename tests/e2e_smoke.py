import os
import sys
import time
import requests


GATEWAY = os.environ.get("GATEWAY_URL", "http://localhost:8000")


def wait_health(path: str, timeout: int = 20):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(path, timeout=3)
            if r.ok:
                return True
        except Exception:
            time.sleep(1)
    return False


def main():
    if not wait_health(f"{GATEWAY}/health"):
        print("Gateway not healthy; is the stack running?", file=sys.stderr)
        sys.exit(1)

    payload = {
        "question": "What was ACME's 2023 revenue?",
        "company": "ACME Corp",
        "year": 2023,
    }
    resp = requests.post(f"{GATEWAY}/qa", json=payload, timeout=60)
    if not resp.ok:
        print(f"QA failed: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    data = resp.json()
    assert data.get("answer"), "No answer returned"
    citations = data.get("citations", [])
    assert citations, "No citations returned"
    print("Smoke OK:", data["answer"][:120], "...", f"citations={len(citations)}", f"eval_passed={data.get('eval', {}).get('passed')}")


if __name__ == "__main__":
    main()

