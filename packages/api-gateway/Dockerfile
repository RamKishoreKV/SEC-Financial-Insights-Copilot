FROM python:3.11-slim

WORKDIR /app

COPY packages/api-gateway/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install shared common package
COPY packages/common /tmp/common
RUN pip install --no-cache-dir /tmp/common

COPY packages/api-gateway/app.py ./app.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

