import argparse
import statistics
import time

import requests


def percentile(values, p):
    values = sorted(values)

    if not values:
        return 0.0

    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    if lower == upper:
        return values[lower]

    weight = index - lower

    return (
        values[lower] * (1.0 - weight)
        + values[upper] * weight
    )


def run_load_test(
    url,
    batch_size,
    requests_count,
):
    payload = {
        "inputs": [
            [0.0] * 1024
            for _ in range(batch_size)
        ]
    }

    latencies_ms = []
    errors = 0

    print("AccelServe HTTP Load Test")
    print("========================")
    print(f"URL: {url}")
    print(f"Batch size: {batch_size}")
    print(f"Requests: {requests_count}")
    print()

    start_total = time.perf_counter()

    for _ in range(requests_count):
        start = time.perf_counter()

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=30,
            )

            end = time.perf_counter()

            if response.status_code != 200:
                errors += 1
                continue

            latency_ms = (
                end - start
            ) * 1000.0

            latencies_ms.append(
                latency_ms
            )

        except requests.RequestException:
            errors += 1

    end_total = time.perf_counter()

    total_seconds = (
        end_total - start_total
    )

    successful = len(
        latencies_ms
    )

    total_samples = (
        successful * batch_size
    )

    request_throughput = (
        successful / total_seconds
        if total_seconds > 0
        else 0.0
    )

    sample_throughput = (
        total_samples / total_seconds
        if total_seconds > 0
        else 0.0
    )

    error_rate = (
        errors / requests_count * 100.0
        if requests_count > 0
        else 0.0
    )

    print("Results")
    print("-------")
    print(
        f"Successful requests: "
        f"{successful}"
    )
    print(
        f"Errors: "
        f"{errors}"
    )
    print(
        f"Error rate: "
        f"{error_rate:.2f}%"
    )

    if latencies_ms:
        print(
            f"Mean latency: "
            f"{statistics.mean(latencies_ms):.2f} ms"
        )
        print(
            f"p50 latency: "
            f"{percentile(latencies_ms, 0.50):.2f} ms"
        )
        print(
            f"p95 latency: "
            f"{percentile(latencies_ms, 0.95):.2f} ms"
        )
        print(
            f"p99 latency: "
            f"{percentile(latencies_ms, 0.99):.2f} ms"
        )

    print(
        f"Request throughput: "
        f"{request_throughput:.2f} req/s"
    )
    print(
        f"Sample throughput: "
        f"{sample_throughput:.2f} samples/s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/infer",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    run_load_test(
        url=args.url,
        batch_size=args.batch_size,
        requests_count=args.requests,
    )