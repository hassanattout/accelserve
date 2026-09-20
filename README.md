# AccelServe

**From CUDA kernels to a continuous-batched AI inference runtime.**

AccelServe is a systems-engineering project for understanding the path between low-level GPU execution and production-style model serving.

The repository now has two complementary tracks:

1. **GPU performance:** C++, CUDA, shared memory, cuBLAS, Tensor Cores, TensorRT, CUDA Graphs, Docker and Kubernetes.
2. **Runtime v2:** an inspectable decoder-only transformer runtime with explicit prefill/decode, K/V caches, logical paged-cache allocation, iteration-level continuous batching, sampling, OpenAI-style serving and Prometheus telemetry.

The goal is not to hide inference behind one framework call. It is to understand where latency, throughput, memory use and scheduling behavior actually come from.

## Runtime v2

Implemented:

- decoder-only transformer written directly in PyTorch
- RMSNorm and causal multi-head attention
- gated SiLU MLP
- explicit prompt prefill
- explicit one-token decode
- per-request tensor K/V caches
- variable-context batched decode
- iteration-level continuous batching
- logical KV block allocator with finite capacity
- deterministic and stochastic sampling
- request cleanup on completion/failure
- OpenAI-style `POST /v1/completions`
- runtime statistics and Prometheus metrics
- CPU correctness and concurrency tests
- concurrent runtime benchmark
- optional Triton RMSNorm experiment

The built-in reference model is deterministic and randomly initialized. It exists for runtime correctness and systems experimentation, not language quality.

## Architecture

```text
Client
  |
  v
/v1/completions
  |
  v
Pending queue
  |
  v
Admission + Prefill ----> per-request KV cache
  |
  v
Active request set
  |
  v
Iteration scheduler
  |
  v
Batched 1-token decode
  |
  v
Sampling
  |
  +--> unfinished -> next decode iteration
  |
  +--> finished -> release KV blocks -> response
```

See [docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch
uvicorn accelserve.api:app --host 0.0.0.0 --port 8000
```

Generate:

```bash
curl -X POST http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "accelserve-tiny",
    "prompt": "hello inference",
    "max_tokens": 16,
    "temperature": 0.0,
    "top_k": 0
  }'
```

Runtime state:

```bash
curl http://127.0.0.1:8000/v1/runtime/stats
```

Prometheus metrics:

```bash
curl http://127.0.0.1:8000/metrics
```

## Benchmark

```bash
PYTHONPATH=. python benchmarks/runtime_benchmark.py \
  --concurrency 8 \
  --max-new-tokens 32
```

The benchmark reports aggregate generated tokens/sec plus mean, p50, p95 and p99 request latency.

## Historical GPU benchmark

On an NVIDIA Tesla T4, batch size 256, 200 runs, FP16:

| Runtime | p50 | p95 | p99 | Throughput |
|---|---:|---:|---:|---:|
| PyTorch FP16 | 0.8684 ms | 0.8827 ms | 0.8913 ms | 294,805.89 samples/s |
| TensorRT FP16 | 0.5819 ms | 0.5933 ms | 0.5961 ms | 439,935.11 samples/s |
| TensorRT + CUDA Graph | 0.5919 ms | 0.6100 ms | 0.6180 ms | 432,514.27 samples/s |

For that fixed MLP workload, TensorRT delivered about **49% higher throughput** than the same-session PyTorch FP16 baseline. These are not LLM-serving results.

## Tests

```bash
PYTHONPATH=. pytest -q
```

The v2 suite checks:

- UTF-8 tokenizer round-trip
- KV allocation, growth and release
- cache-capacity enforcement
- prefill/decode cache growth
- concurrent request handling
- sequence-capacity validation
- OpenAI-style API serving
- Prometheus telemetry
- batched-decode equivalence against independent decode

## Existing GPU track

The repository also preserves the earlier performance work:

- C++ CPU baselines
- CUDA vector and GEMM kernels
- CUDA Events
- host/device transfer analysis
- GPU data residency
- shared-memory tiling
- cuBLAS comparisons
- FP16 / Tensor Core benchmarks
- PyTorch GPU inference
- ONNX and ModelOpt
- TensorRT engine build/execution
- CUDA Graph experiments
- FastAPI serving
- Docker image optimization
- Kubernetes CPU/GPU manifests

## Roadmap

1. Real open-weight Qwen/Gemma/Llama-compatible model adapter while preserving AccelServe-owned scheduling.
2. Physical paged KV cache with preallocated K/V slabs and block tables.
3. Measured Triton RMSNorm, RoPE, gated-activation and decode-attention kernels.
4. Prefix caching and TTFT/cache-hit benchmarks.
5. Speculative decoding.
6. Multi-GPU NCCL tensor-parallel serving.
7. Multimodal/VLA extension for sensor-to-action inference latency.

## Engineering principles

- Benchmark end-to-end, not only kernels.
- Keep performance claims tied to reproducible workloads.
- Use vendor libraries when they are better, but understand the lower-level primitive.
- Keep memory and scheduling visible as first-class inference problems.
- Do not claim production readiness for a reference implementation.

## Current limitations

Runtime v2 is a reference system, not a production LLM server. The model is random, the tensor KV cache is not physically paged yet, prefill is not batched, and distributed inference is not implemented. Those are explicit next milestones rather than hidden behind a framework abstraction.

## Author

**Hassan Attout**

AI Systems, GPU Computing and Physical AI Infrastructure
