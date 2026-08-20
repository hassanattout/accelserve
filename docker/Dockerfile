FROM python:3.12-slim

WORKDIR /app

COPY requirements-cpu.txt .

RUN pip install --no-cache-dir -r requirements-cpu.txt

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

COPY api ./api
COPY inference ./inference

RUN useradd --create-home --uid 10001 accelserve
USER accelserve

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python3", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
