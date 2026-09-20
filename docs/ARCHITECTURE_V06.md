# AccelServe v0.6 Architecture

AccelServe v0.6 separates request scheduling, model execution, and physical K/V memory.

## Data path

```text
request
  -> pending queue
  -> prefill
  -> physical paged KV
  -> active request set
  -> batched one-token decode
  -> sampling
  -> response / page release
```

The cache is preallocated as:

```text
[layer, physical_block, kv_head, token_in_block, head_dim]
```

Each request owns a block table. K/V writes go directly into the backing slabs. Releasing a request returns its blocks to the free list without reallocating the slabs.

The PyTorch reference attention path currently gathers pages into contiguous tensors. A direct Triton/CUDA paged-attention kernel that consumes the block table is the next performance milestone.

## Qwen2.5 path

`load_qwen2_pretrained()` maps Qwen2/Qwen2.5 checkpoint weights into AccelServe's own decoder implementation.

Transformers is used for checkpoint/tokenizer loading only. AccelServe owns:
- forward execution
- GQA and RoPE
- physical K/V cache
- continuous batching
- token-by-token decode
- sampling and request lifecycle

Default target:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

## Evidence

Use:
- `benchmarks/validate_qwen2_parity.py` for logit parity
- `benchmarks/open_weight_benchmark.py` for same-host comparisons
- `profiling/profile_qwen_runtime.py` for Nsight/NVTX profiling

No Qwen performance numbers are hard-coded; results must be generated on the named hardware/software stack.
