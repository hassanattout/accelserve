# AccelServe HTTP Load Test Results

## Environment

- Runtime: FastAPI + Uvicorn
- Backend: PyTorch
- Device: CPU
- Client: Python `requests`
- Requests per benchmark: 100
- Load pattern: sequential HTTP requests
- Endpoint: `/infer`

## Results

| Batch Size | Mean Latency | p50 | p95 | p99 | Request Throughput | Sample Throughput | Error Rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.75 ms | 9.62 ms | 11.68 ms | 13.51 ms | 102.54 req/s | 102.54 samples/s | 0.00% |
| 8 | 19.70 ms | 19.34 ms | 22.99 ms | 24.24 ms | 50.74 req/s | 405.94 samples/s | 0.00% |
| 32 | 48.50 ms | 45.71 ms | 69.45 ms | 74.43 ms | 20.62 req/s | 659.70 samples/s | 0.00% |

## Observations

Increasing batch size increased end-to-end request latency while significantly increasing sample throughput.

Batch size 1 achieved the lowest latency:

- p50: 9.62 ms
- p95: 11.68 ms
- p99: 13.51 ms

Batch size 32 achieved the highest sample throughput:

- 659.70 samples/s

Compared with batch size 1, batch size 32 delivered approximately 6.4x higher sample throughput while increasing p50 request latency from 9.62 ms to 45.71 ms.

All 300 requests across the three benchmark configurations completed successfully with a 0.00% error rate.

## Interpretation

These measurements demonstrate the latency-throughput tradeoff inherent in batched inference.

Smaller batches prioritize individual request latency.

Larger batches amortize model and serving overhead across more samples, increasing total throughput at the cost of higher request latency.

These results represent a sequential single-client CPU serving workload and should not be interpreted as maximum server capacity.

Concurrent load testing is required to evaluate behavior under multiple simultaneous requests.