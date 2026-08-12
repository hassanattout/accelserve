import time
import random
import statistics
import requests

URL = "http://127.0.0.1:8000/infer"

INPUT_DIM = 1024
BATCH_SIZES = [1, 8, 32]

RUNS = 10


def make_payload(batch_size):
    return {
        "inputs": [
            [
                random.uniform(-1.0, 1.0)
                for _ in range(INPUT_DIM)
            ]
            for _ in range(batch_size)
        ]
    }


print("AccelServe HTTP Batch Benchmark")
print("===============================")

for batch_size in BATCH_SIZES:

    model_times = []
    http_times = []

    payload = make_payload(batch_size)

    for _ in range(RUNS):

        start = time.perf_counter()

        response = requests.post(
            URL,
            json=payload,
            timeout=60
        )

        end = time.perf_counter()

        response.raise_for_status()

        result = response.json()

        model_times.append(
            result["latency_ms"]
        )

        http_times.append(
            (end - start) * 1000.0
        )

    model_median = statistics.median(
        model_times
    )

    http_median = statistics.median(
        http_times
    )

    overhead = (
        http_median - model_median
    )

    throughput = (
        batch_size
        /
        (http_median / 1000.0)
    )

    print()
    print(f"Batch size: {batch_size}")
    print(
        f"Model median:      "
        f"{model_median:.2f} ms"
    )
    print(
        f"HTTP median:       "
        f"{http_median:.2f} ms"
    )
    print(
        f"Serving overhead:  "
        f"{overhead:.2f} ms"
    )
    print(
        f"HTTP throughput:   "
        f"{throughput:.2f} samples/s"
    )