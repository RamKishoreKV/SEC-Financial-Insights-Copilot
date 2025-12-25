FROM python:3.11-slim

WORKDIR /app

COPY packages/orchestrator/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY packages/common /tmp/common
RUN pip install --no-cache-dir /tmp/common

COPY packages/orchestrator/app.py ./app.py

EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]

