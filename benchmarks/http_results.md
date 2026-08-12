# AccelServe HTTP Serving Benchmark

## Environment

- Device: CPU
- Platform: local Mac
- API: FastAPI
- Transport: HTTP/JSON
- Runs per batch size: 10

## Results

| Batch | Model Median | HTTP Median | Serving Overhead | HTTP Throughput |
|---:|---:|---:|---:|---:|
| 1 | 3.44 ms | 11.16 ms | 7.72 ms | 89.60 samples/s |
| 8 | 8.24 ms | 28.39 ms | 20.15 ms | 281.76 samples/s |
| 32 | 13.81 ms | 77.76 ms | 63.95 ms | 411.53 samples/s |

## Observations

Batching improved end-to-end throughput, but HTTP/JSON overhead increased substantially with larger numerical payloads.

At batch size 32, model execution accounted for only a fraction of total request latency, showing that serialization, validation, tensor construction, and response generation can dominate serving latency.

Single-request timings were substantially higher than warmed median measurements, reinforcing the importance of warmup runs and repeated measurements.