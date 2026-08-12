FROM python:3.12-slim

WORKDIR /app

COPY requirements-cpu.txt .

RUN pip install --no-cache-dir -r requirements-cpu.txt

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

COPY api ./api

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]