import time
import statistics
import torch
import torch.nn as nn
import tensorrt as trt

torch.manual_seed(0)

DEVICE = "cuda:0"

BATCH_SIZE = 256
INPUT_DIM = 1024
HIDDEN_DIM = 4096
OUTPUT_DIM = 1000

WARMUP_RUNS = 20
BENCHMARK_RUNS = 200

ENGINE_PATH = "inference/accelserve_mlp_fp16_native_io.engine"


# ============================================================
# Helpers
# ============================================================

def percentile(values, q):
    values = sorted(values)

    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    if lower == upper:
        return values[lower]

    fraction = index - lower

    return (
        values[lower] * (1.0 - fraction)
        + values[upper] * fraction
    )


def summarize(name, times_ms):
    p50 = percentile(times_ms, 0.50)
    p95 = percentile(times_ms, 0.95)
    p99 = percentile(times_ms, 0.99)

    throughput = (
        BATCH_SIZE
        /
        (p50 / 1000.0)
    )

    return {
        "name": name,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "throughput": throughput
    }


# ============================================================
# Model
# ============================================================

class InferenceMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        )

    def forward(self, x):
        return self.net(x)


model = (
    InferenceMLP()
    .eval()
    .to(DEVICE)
    .half()
)

x = torch.randn(
    BATCH_SIZE,
    INPUT_DIM,
    device=DEVICE,
    dtype=torch.float16
)


# ============================================================
# PyTorch FP16
# ============================================================

with torch.inference_mode():

    for _ in range(WARMUP_RUNS):
        _ = model(x)

torch.cuda.synchronize()

start_event = torch.cuda.Event(enable_timing=True)
stop_event = torch.cuda.Event(enable_timing=True)

pytorch_times = []

with torch.inference_mode():

    for _ in range(BENCHMARK_RUNS):

        start_event.record()

        y_pytorch = model(x)

        stop_event.record()

        torch.cuda.synchronize()

        pytorch_times.append(
            start_event.elapsed_time(stop_event)
        )


# ============================================================
# TensorRT setup
# ============================================================

logger = trt.Logger(trt.Logger.WARNING)

with open(ENGINE_PATH, "rb") as f:
    engine_bytes = f.read()

runtime = trt.Runtime(logger)

engine = runtime.deserialize_cuda_engine(
    engine_bytes
)

if engine is None:
    raise RuntimeError(
        "TensorRT engine deserialization failed"
    )

context = engine.create_execution_context()

if context is None:
    raise RuntimeError(
        "TensorRT context creation failed"
    )

trt_output = torch.empty(
    BATCH_SIZE,
    OUTPUT_DIM,
    device=DEVICE,
    dtype=torch.float16
)

context.set_tensor_address(
    "input",
    x.data_ptr()
)

context.set_tensor_address(
    "output",
    trt_output.data_ptr()
)

stream = torch.cuda.Stream()


# ============================================================
# TensorRT warmup
# ============================================================

with torch.cuda.stream(stream):

    for _ in range(WARMUP_RUNS):

        ok = context.execute_async_v3(
            stream_handle=stream.cuda_stream
        )

        if not ok:
            raise RuntimeError(
                "TensorRT warmup failed"
            )

stream.synchronize()


# ============================================================
# TensorRT regular execution
# ============================================================

trt_times = []

start_event_trt = torch.cuda.Event(
    enable_timing=True
)

stop_event_trt = torch.cuda.Event(
    enable_timing=True
)

for _ in range(BENCHMARK_RUNS):

    with torch.cuda.stream(stream):

        start_event_trt.record(stream)

        ok = context.execute_async_v3(
            stream_handle=stream.cuda_stream
        )

        stop_event_trt.record(stream)

    if not ok:
        raise RuntimeError(
            "TensorRT inference failed"
        )

    stream.synchronize()

    trt_times.append(
        start_event_trt.elapsed_time(
            stop_event_trt
        )
    )


# ============================================================
# Capture TensorRT CUDA Graph
# ============================================================

graph = torch.cuda.CUDAGraph()

with torch.cuda.graph(
    graph,
    stream=stream
):

    ok = context.execute_async_v3(
        stream_handle=stream.cuda_stream
    )

    if not ok:
        raise RuntimeError(
            "CUDA Graph capture failed"
        )

torch.cuda.synchronize()


# ============================================================
# TensorRT CUDA Graph benchmark
# ============================================================

graph_times = []

start_event_graph = torch.cuda.Event(
    enable_timing=True
)

stop_event_graph = torch.cuda.Event(
    enable_timing=True
)

for _ in range(BENCHMARK_RUNS):

    start_event_graph.record()

    graph.replay()

    stop_event_graph.record()

    torch.cuda.synchronize()

    graph_times.append(
        start_event_graph.elapsed_time(
            stop_event_graph
        )
    )


# ============================================================
# Numerical comparison
# ============================================================

with torch.inference_mode():
    y_reference = model(x)

# Make sure TensorRT output corresponds to the same input
with torch.cuda.stream(stream):

    ok = context.execute_async_v3(
        stream_handle=stream.cuda_stream
    )

stream.synchronize()

if not ok:
    raise RuntimeError(
        "Final TensorRT inference failed"
    )

max_abs_error = (
    y_reference.float()
    - trt_output.float()
).abs().max().item()

mean_abs_error = (
    y_reference.float()
    - trt_output.float()
).abs().mean().item()


# ============================================================
# Summaries
# ============================================================

results = [
    summarize(
        "PyTorch FP16",
        pytorch_times
    ),

    summarize(
        "TensorRT FP16",
        trt_times
    ),

    summarize(
        "TensorRT + CUDA Graph",
        graph_times
    )
]


# ============================================================
# Print results
# ============================================================

print()
print("AccelServe Unified Inference Benchmark")
print("======================================")

print(
    f"GPU: {torch.cuda.get_device_name(0)}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Runs: {BENCHMARK_RUNS}"
)

print()

print(
    f"{'Runtime':<24}"
    f"{'p50 ms':>10}"
    f"{'p95 ms':>10}"
    f"{'p99 ms':>10}"
    f"{'samples/s':>16}"
)

print("-" * 70)

for result in results:

    print(
        f"{result['name']:<24}"
        f"{result['p50']:>10.4f}"
        f"{result['p95']:>10.4f}"
        f"{result['p99']:>10.4f}"
        f"{result['throughput']:>16.2f}"
    )

print()

print(
    f"TensorRT max absolute error: "
    f"{max_abs_error:.8f}"
)

print(
    f"TensorRT mean absolute error: "
    f"{mean_abs_error:.8f}"
)