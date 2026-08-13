import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def send_request(url, payload):
    start = time.perf_counter()

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=30,
        )

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000.0

        if response.status_code != 200:
            return False, latency_ms

        return True, latency_ms

    except requests.RequestException:
        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000.0

        return False, latency_ms


def run_load_test(
    url,
    batch_size,
    requests_count,
    concurrency,
):
    payload = {
        "inputs": [
            [0.0] * 1024
            for _ in range(batch_size)
        ]
    }

    latencies_ms = []
    errors = 0

    print("AccelServe Concurrent HTTP Load Test")
    print("===================================")
    print(f"URL: {url}")
    print(f"Batch size: {batch_size}")
    print(f"Requests: {requests_count}")
    print(f"Concurrency: {concurrency}")
    print()

    start_total = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = [
            executor.submit(
                send_request,
                url,
                payload,
            )
            for _ in range(requests_count)
        ]

        for future in as_completed(futures):
            success, latency_ms = future.result()

            if success:
                latencies_ms.append(
                    latency_ms
                )
            else:
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

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    run_load_test(
        url=args.url,
        batch_size=args.batch_size,
        requests_count=args.requests,
        concurrency=args.concurrency,
    )