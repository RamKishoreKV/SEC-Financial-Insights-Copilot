# Tests

Planned e2e smoke (to be added):
- Seed corpus (already bundled).
- Start stack with `docker compose -f infra/docker-compose.yaml up --build`.
- Hit `POST http://localhost:8000/qa` with a revenue question and assert answer + citations + eval flags.

New e2e smoke (manual):
1) Ensure stack is running (`docker compose -f infra/docker-compose.yaml up -d`).
2) Run `python tests/e2e_smoke.py` (requires `requests` installed). It queries `/health` then `/qa` and checks for answer + citations.

