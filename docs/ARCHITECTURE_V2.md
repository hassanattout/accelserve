# AccelServe v2 Architecture

AccelServe v2 is a reference inference runtime designed to expose the mechanics that are usually hidden behind a high-level `generate()` call.

The runtime separates **prefill**, **decode**, **request scheduling**, **KV-cache capacity management**, **sampling**, and **HTTP serving**.

## Data path

```text
HTTP request
   |
   v
FastAPI /v1/completions
   |
   v
Pending request queue
   |
   v
Admission + prefill
   |        \
   |         -> per-request tensor KV cache
   v
Active request set
   |
   v
Iteration-level scheduler
   |
   v
Batched one-token decode
   |
   v
Sampling
   |
   +--> unfinished -> next scheduler iteration
   |
   +--> finished -> release KV blocks -> response
```

## Reference model

`accelserve/model.py` contains a small decoder-only transformer implemented directly in PyTorch with token embeddings, learned positions, RMSNorm, multi-head causal attention, gated SiLU feed-forward layers, tied LM-head weights, explicit prefill, explicit token decode, per-request K/V tensors, and variable-context batched decode.

The built-in model is deterministic and randomly initialized. It exists for runtime correctness and systems experimentation, not language quality.

## Continuous batching

`ContinuousBatchingEngine` keeps pending and active request populations. Every iteration it admits work, performs decode for the active batch, removes completed requests immediately, releases their KV allocation, and makes capacity available to waiting requests.

## KV cache

The current implementation has real K/V tensors per request plus a logical block allocator that enforces a finite cache budget. The next performance milestone is a physically paged K/V slab with per-request block tables, removing repeated `torch.cat()` growth.

## Telemetry

Endpoints:

```text
GET  /health
GET  /v1/runtime/stats
GET  /metrics
POST /v1/completions
```

Prometheus metrics expose request counts, generated tokens, request latency, queue/prefill latency, active and pending requests, and logical KV-cache utilization.

## Current limitations

- randomly initialized reference language model
- per-request tensor KV storage rather than physical paged attention
- request-by-request prefill
- no prefix cache yet
- no speculative decoding
- no tensor/pipeline parallelism
- no NCCL worker group
- no CUDA Graph capture in v2
- no production tokenizer/model adapter yet
- no Triton attention kernel yet

## Next milestones

1. Add a small real Qwen/Gemma/Llama-compatible model adapter while keeping AccelServe-owned scheduling.
2. Replace logical pages with a preallocated physical KV cache and block tables.
3. Add measured Triton kernels for RMSNorm, RoPE, gated activation and decode attention.
4. Add prefix caching and benchmark TTFT/cache-hit effects.
5. Add speculative decoding.
6. Add multi-process NCCL tensor parallelism, request routing and worker health.
7. Extend the runtime to multimodal/VLA workloads and optimize sensor-to-action latency.
