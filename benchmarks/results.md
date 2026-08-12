# AccelServe Inference Benchmark Results

## Environment

- GPU: NVIDIA Tesla T4
- Batch size: 256
- Benchmark runs: 200
- Precision: FP16
- PyTorch: 2.10.0+cu128
- TensorRT: 11.2.1.2

## Unified Inference Benchmark

| Runtime | p50 latency | p95 latency | p99 latency | Throughput |
|---|---:|---:|---:|---:|
| PyTorch FP16 | 0.8684 ms | 0.8827 ms | 0.8913 ms | 294,805.89 samples/s |
| TensorRT FP16 | 0.5819 ms | 0.5933 ms | 0.5961 ms | 439,935.11 samples/s |
| TensorRT + CUDA Graph | 0.5919 ms | 0.6100 ms | 0.6180 ms | 432,514.27 samples/s |

## Key Result

TensorRT FP16 achieved approximately 1.49x lower p50 latency than PyTorch FP16 in the same-session benchmark.

TensorRT throughput was approximately 49% higher than PyTorch FP16.

CUDA Graph replay did not improve performance for this workload.

## Numerical Difference

- Maximum absolute error: 0.00244141
- Mean absolute error: 0.00033709

Note: strict numerical validation should use identical persisted model weights for both PyTorch and TensorRT.