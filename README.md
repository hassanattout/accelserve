# AccelServe

**A from-scratch continuous-batching LLM runtime with physical paged K/V memory.**

AccelServe is a systems-engineering project that connects low-level CUDA/TensorRT work to the internals of modern LLM serving.

## v0.6

The runtime now implements:

- preallocated **physical K/V slabs**
- per-request block tables and page reuse
- continuous batching with explicit prefill/decode
- grouped-query attention (GQA)
- RoPE
- RMSNorm + gated SiLU MLP
- real **Qwen2/Qwen2.5 checkpoint loading**
- AccelServe-owned forward execution and token scheduling
- TTFT, TPOT, p50/p95/p99 and token-throughput measurement
- Hugging Face and optional vLLM comparison harness
- Qwen2.5 logit-parity validation
- Nsight/NVTX profiling workload
- optional Triton RMSNorm and RoPE experiments
- Prometheus request/cache metrics

Transformers is used only to download/deserialise Qwen weights and tokenize text. AccelServe owns the model execution path, scheduler, cache and decoding loop.

Default open-weight target:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

## Architecture

```text
requests
   |
   v
pending queue
   |
   v
prefill -------------------------+
   |                             |
   v                             v
active request set        physical paged KV
   |                     [layer, block,
   |                      kv_head, token, dim]
   v
iteration scheduler
   |
   v
batched one-token decode
   |
   v
GQA + RoPE
   |
   v
sampling
   |
   +---- unfinished -> next iteration
   |
   +---- finished -> return blocks -> response
```

See `docs/ARCHITECTURE_V06.md`.

## Physical paged KV cache

v0.5 tracked pages logically but still grew per-request tensors. v0.6 preallocates the actual K/V backing tensors once.

A request owns a block table such as:

```text
request A -> [3, 17, 8]
request B -> [0, 11]
```

K/V values are written directly into those physical pages. When a request finishes, its pages return to the allocator without reallocating the backing slabs.

The current PyTorch attention path still gathers pages into contiguous tensors before attention. A direct block-table Triton/CUDA paged-attention kernel is the next performance step.

## Reference runtime

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch
uvicorn accelserve.api:app --port 8000
```

## Run Qwen2.5 through AccelServe

```bash
pip install -e '.[open-weight]'

ACCELSERVE_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
ACCELSERVE_DEVICE=cuda \
uvicorn accelserve.api:app --host 0.0.0.0 --port 8000
```

Then:

```bash
curl -X POST http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Explain continuous batching in one paragraph.",
    "max_tokens": 64,
    "temperature": 0
  }'
```

The response exposes TTFT, TPOT, total generation time, end-to-end latency, device and KV-cache mode.

## Validate Qwen weight mapping

Before using performance results, validate the custom execution path against Transformers:

```bash
python benchmarks/validate_qwen2_parity.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda
```

It reports maximum/mean logit error and next-token argmax agreement. No parity numbers are hard-coded into the repository.

## Compare AccelServe, Transformers and vLLM

```bash
python benchmarks/open_weight_benchmark.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --backends accelserve transformers \
  --requests 4 \
  --max-tokens 32 \
  --device cuda
```

On a compatible GPU host with vLLM installed:

```bash
python benchmarks/open_weight_benchmark.py \
  --backends accelserve transformers vllm \
  --device cuda
```

Results are generated on the active host; the repo does not claim universal LLM-serving wins.

## Runtime benchmark

```bash
PYTHONPATH=. python benchmarks/runtime_benchmark.py \
  --concurrency 8 \
  --max-new-tokens 32 \
  --device cuda \
  --json
```

Metrics include TTFT, TPOT, p50/p95/p99 request latency, generated tokens/sec and physical K/V bytes reserved.

## Triton experiments

Install Triton separately on a supported NVIDIA/Linux environment, then run:

```bash
python benchmarks/triton_kernel_benchmark.py
```

The benchmark validates numerical error and compares the existing Triton RMSNorm plus the v0.6 Triton RoPE experiment against PyTorch reference implementations.

## GPU profiling

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  -o accelserve_v06 \
  python profiling/profile_qwen_runtime.py
```

The workload emits an NVTX range around the Qwen request batch.

## Historical GPU track

The repository also preserves:

- custom C++/CUDA kernels
- transfer-vs-compute analysis
- shared-memory GEMM
- cuBLAS
- Tensor Cores / FP16
- PyTorch inference
- ONNX / ModelOpt
- TensorRT
- CUDA Graphs
- FastAPI
- Docker
- Kubernetes

Historical Tesla T4 fixed-MLP benchmark:

| Runtime | p50 | Throughput |
|---|---:|---:|
| PyTorch FP16 | 0.8684 ms | 294,805.89 samples/s |
| TensorRT FP16 | 0.5819 ms | 439,935.11 samples/s |

Those numbers are **not** Qwen/LLM-serving results.

## Tests

```bash
pytest -q
```

CPU CI covers page allocation, physical K/V writes/reads, fixed backing storage, GQA/RoPE execution, variable-context batched decode, concurrent scheduling, cache cleanup, API metrics and TTFT/TPOT reporting.

## Next performance milestones

1. Direct block-table **paged-attention Triton/CUDA kernel**
2. Integrate measured Triton hot-path kernels
3. batched prefill
4. prefix caching
5. CUDA Graph decode path
6. speculative decoding
7. INT8/FP8/KV-cache quantization experiments
8. NCCL tensor parallelism and multi-GPU worker orchestration
9. multimodal/VLA serving for Physical AI

## Scope

AccelServe is a serious reference implementation and performance-learning platform, not a production replacement for vLLM/SGLang/TensorRT-LLM. Performance claims should come from reproducible benchmarks on named hardware.

## Author

**Hassan Attout**

AI Systems • GPU Computing • Inference Infrastructure • Physical AI
