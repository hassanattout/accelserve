import time
import requests
import random

URL = "http://127.0.0.1:8000/infer"

BATCH_SIZE = 1
INPUT_DIM = 1024

payload = {
    "inputs": [
        [
            random.uniform(-1.0, 1.0)
            for _ in range(INPUT_DIM)
        ]
        for _ in range(BATCH_SIZE)
    ]
}

start = time.perf_counter()

response = requests.post(
    URL,
    json=payload,
    timeout=30
)

end = time.perf_counter()

response.raise_for_status()

result = response.json()

http_latency_ms = (
    end - start
) * 1000.0

print("AccelServe HTTP Inference Test")
print("==============================")
print("Status:", response.status_code)
print("Batch size:", result["batch_size"])
print("Model latency:", round(result["latency_ms"], 4), "ms")
print("HTTP end-to-end latency:", round(http_latency_ms, 4), "ms")
print("Device:", result["device"])
print("Output length:", len(result["outputs"][0]))