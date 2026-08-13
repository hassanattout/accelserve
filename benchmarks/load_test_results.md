# AccelServe HTTP Load Test Results

## Environment

- Runtime: FastAPI + Uvicorn
- Backend: PyTorch
- Device: CPU
- Client: Python `requests`
- Endpoint: `/infer`
- Sequential requests per configuration: 100
- Concurrent requests per configuration: 200

---

## Sequential Baseline

| Batch | Mean | p50 | p95 | p99 | Req/s | Samples/s | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.75 ms | 9.62 ms | 11.68 ms | 13.51 ms | 102.54 | 102.54 | 0.00% |
| 8 | 19.70 ms | 19.34 ms | 22.99 ms | 24.24 ms | 50.74 | 405.94 | 0.00% |
| 32 | 48.50 ms | 45.71 ms | 69.45 ms | 74.43 ms | 20.62 | 659.70 | 0.00% |

Increasing batch size increased request latency while substantially improving sample throughput.

---

## Concurrency 4

| Batch | Mean | p50 | p95 | p99 | Req/s | Samples/s | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29.85 ms | 24.61 ms | 50.69 ms | 173.58 ms | 131.28 | 131.28 | 0.00% |
| 8 | 50.30 ms | 49.62 ms | 65.80 ms | 77.10 ms | 78.73 | 629.86 | 0.00% |
| 32 | 115.59 ms | 112.42 ms | 150.11 ms | 202.42 ms | 34.39 | 1100.62 | 0.00% |

Concurrency improved throughput significantly compared with sequential execution.

Batch size 32 achieved the highest measured CPU sample throughput:

**1100.62 samples/s**

---

## Concurrency 8

| Batch | Mean | p50 | p95 | p99 | Req/s | Samples/s | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 38.69 ms | 39.68 ms | 51.89 ms | 58.00 ms | 200.05 | 200.05 | 0.00% |
| 8 | 93.38 ms | 91.91 ms | 119.86 ms | 132.48 ms | 84.59 | 676.71 | 0.00% |
| 32 | 292.07 ms | 267.33 ms | 608.79 ms | 650.30 ms | 27.20 | 870.44 | 0.00% |

---

## Concurrency Scaling

### Batch 1

```text
Sequential     102.54 samples/s
Concurrency 4  131.28 samples/s
Concurrency 8  200.05 samples/s
```

The small workload continued to benefit from additional concurrency.

### Batch 8

```text
Sequential     405.94 samples/s
Concurrency 4  629.86 samples/s
Concurrency 8  676.71 samples/s
```

Throughput continued to increase, but the improvement from concurrency 4 to 8 was small relative to the increase in latency.

This indicates the workload was approaching CPU saturation.

### Batch 32

```text
Sequential      659.70 samples/s
Concurrency 4  1100.62 samples/s
Concurrency 8   870.44 samples/s
```

Concurrency 4 provided the highest throughput.

Increasing concurrency from 4 to 8 reduced throughput by approximately 21% while significantly increasing tail latency.

This indicates that the CPU serving path became overloaded for the batch-32 workload.

---

## Peak Observed Throughput

The highest sample throughput measured during these tests was:

```text
1100.62 samples/s
```

Configuration:

```text
Batch size:   32
Concurrency:  4
```

The highest request throughput measured was:

```text
200.05 requests/s
```

Configuration:

```text
Batch size:   1
Concurrency:  8
```

---

## Key Findings

### Batching improves sample throughput

Larger batches amortize inference and serving overhead across more samples.

### Concurrency improves utilization until saturation

Moderate concurrency improved throughput substantially.

However, increasing concurrency beyond the useful saturation point increased queueing and contention without increasing useful work.

### The optimal concurrency depends on workload size

Small batch-size requests continued scaling at concurrency 8.

Batch size 8 showed diminishing returns.

Batch size 32 performed best at concurrency 4 and degraded at concurrency 8.

### Tail latency exposes saturation

Mean and median latency alone do not fully describe serving behavior.

At batch size 32 and concurrency 8:

```text
p50: 267.33 ms
p95: 608.79 ms
p99: 650.30 ms
```

This large tail-latency increase accompanied a reduction in throughput, indicating overload.

### Error rate alone does not indicate system health

All requests completed successfully with a 0.00% error rate, even when throughput degraded and latency increased substantially.

A production inference system therefore needs latency and saturation metrics in addition to error monitoring.

---

## Conclusion

AccelServe's CPU load tests demonstrate three core inference-serving tradeoffs:

```text
batch size
↕
latency vs throughput

concurrency
↕
utilization vs contention

load
↕
throughput vs tail latency
```

The measurements show that increasing parallelism improves throughput only until the serving stack reaches its saturation point.

For the tested CPU workload, the highest observed sample throughput occurred at:

```text
Batch size 32
Concurrency 4
1100.62 samples/s
```

Increasing the same workload to concurrency 8 reduced throughput and sharply increased tail latency.

These results are workload- and hardware-specific and should not be interpreted as universal performance limits.