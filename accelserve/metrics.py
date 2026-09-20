from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter(
    "accelserve_v2_requests_total",
    "Completed AccelServe v2 generation requests",
    ["status"],
)
GENERATED_TOKENS = Counter(
    "accelserve_v2_generated_tokens_total",
    "Generated tokens returned by AccelServe v2",
)
REQUEST_LATENCY = Histogram(
    "accelserve_v2_request_latency_seconds",
    "End-to-end generation request latency",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
QUEUE_LATENCY = Histogram(
    "accelserve_v2_queue_latency_seconds",
    "Request queue and prefill latency before generation starts",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 5),
)
ACTIVE_REQUESTS = Gauge(
    "accelserve_v2_active_requests",
    "Currently active generation requests",
)
PENDING_REQUESTS = Gauge(
    "accelserve_v2_pending_requests",
    "Queued generation requests waiting for admission",
)
KV_UTILIZATION = Gauge(
    "accelserve_v2_kv_cache_utilization_ratio",
    "Logical KV-cache block utilization ratio",
)


def sync_runtime_gauges(stats: dict[str, int | float | str | bool]) -> None:
    ACTIVE_REQUESTS.set(float(stats["active_requests"]))
    PENDING_REQUESTS.set(float(stats["pending_requests"]))
    KV_UTILIZATION.set(float(stats["kv_utilization"]))
