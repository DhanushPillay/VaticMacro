"""Sandbox p95 latency."""

import time

from app.factory import create_app

app = create_app()
c = app.test_client()
times = []
for _ in range(20):
    t0 = time.perf_counter()
    c.post("/api/predictive-sandbox", json={"WPIATT01INM661N_pct_1m": 1.0})
    times.append((time.perf_counter() - t0) * 1000)
times.sort()
p95 = times[int(0.95 * len(times))]
print(
    f"sandbox p50={times[len(times) // 2]:.1f}ms p95={p95:.1f}ms — {'PASS' if p95 < 100 else 'SLOW'}"
)
