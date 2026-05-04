#!/usr/bin/env python3
import argparse
import json
import statistics
import subprocess
import time
import urllib.request
from typing import Any


def post_json(url: str, payload: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, timeout: int = 60) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def clear_l1(http_port: int) -> float:
    url = f"http://localhost:{http_port}/api/clear-cache"
    t0 = time.perf_counter()
    post_json(url, {}, timeout=120)
    return (time.perf_counter() - t0) * 1000


def l1_object_count(http_port: int) -> int:
    status = get_json(f"http://localhost:{http_port}/api/status")
    return int(status["storage_manager"]["l1_manager"]["total_object_count"])


def wait_l1_empty(http_port: int, timeout_s: float = 30.0) -> None:
    deadline = time.perf_counter() + timeout_s
    last_count = -1
    while time.perf_counter() < deadline:
        last_count = l1_object_count(http_port)
        if last_count == 0:
            return
        time.sleep(0.2)
    raise TimeoutError(f"L1 did not become empty; last total_object_count={last_count}")


def completion(vllm_port: int, model: str, prompt: str, max_tokens: int) -> tuple[float, dict[str, Any]]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    t0 = time.perf_counter()
    out = post_json(f"http://localhost:{vllm_port}/v1/completions", payload, timeout=300)
    dt = (time.perf_counter() - t0) * 1000
    return dt, out


def count_log_pattern(log_path: str, pattern: str) -> int:
    try:
        out = subprocess.check_output(
            ["bash", "-lc", f"grep -c {json.dumps(pattern)} {json.dumps(log_path)} || true"],
            text=True,
        ).strip()
        return int(out or "0")
    except Exception:
        return 0


def tail_matching_lines(log_path: str, pattern: str, n: int = 20) -> list[str]:
    try:
        out = subprocess.check_output(
            [
                "bash",
                "-lc",
                f"grep -n {json.dumps(pattern)} {json.dumps(log_path)} | tail -{n}",
            ],
            text=True,
        )
        return [x for x in out.splitlines() if x.strip()]
    except Exception:
        return []


def summarize(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    xs_sorted = sorted(xs)
    return {
        "count": len(xs),
        "mean_ms": statistics.mean(xs),
        "median_ms": statistics.median(xs),
        "min_ms": min(xs),
        "max_ms": max(xs),
        "p90_ms": xs_sorted[int(0.9 * (len(xs_sorted) - 1))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--lmcache-http-port", type=int, default=8080)
    parser.add_argument("--lmcache-log", required=True)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--post-cold-sleep", type=float, default=1.0)
    parser.add_argument(
        "--prompt",
        default=(
            "Write a short story about a student learning distributed systems. "
            "The story should mention cache, storage, and scheduling. "
        ) * 32,
    )
    args = parser.parse_args()

    status = get_json(f"http://localhost:{args.lmcache_http_port}/api/status")
    print("LMCache healthy:", status.get("is_healthy"))
    print("chunk_size:", status.get("chunk_size"))

    cold_latencies = []
    clear_latencies = []
    l2_latencies = []

    before_prefetch = count_log_pattern(args.lmcache_log, "Prefetch request completed")
    before_retrieved = count_log_pattern(args.lmcache_log, "Retrieved")

    rows = []

    for i in range(args.iters):
        print(f"\n=== Iteration {i + 1}/{args.iters} ===")

        cold_ms, cold_out = completion(
            args.vllm_port, args.model, args.prompt, args.max_tokens
        )
        cold_latencies.append(cold_ms)
        print(f"cold_ms={cold_ms:.2f}")

        if args.post_cold_sleep > 0:
            time.sleep(args.post_cold_sleep)

        clear_ms = clear_l1(args.lmcache_http_port)
        wait_l1_empty(args.lmcache_http_port)
        clear_latencies.append(clear_ms)
        print(f"clear_l1_ms={clear_ms:.2f}, l1_object_count=0")

        l2_ms, l2_out = completion(
            args.vllm_port, args.model, args.prompt, args.max_tokens
        )
        l2_latencies.append(l2_ms)
        print(f"l2_hit_ms={l2_ms:.2f}")

        cold_text = cold_out["choices"][0]["text"]
        l2_text = l2_out["choices"][0]["text"]

        rows.append(
            {
                "iter": i + 1,
                "cold_ms": cold_ms,
                "clear_l1_ms": clear_ms,
                "l2_hit_ms": l2_ms,
                "cold_chars": len(cold_text),
                "l2_chars": len(l2_text),
            }
        )

    after_prefetch = count_log_pattern(args.lmcache_log, "Prefetch request completed")
    after_retrieved = count_log_pattern(args.lmcache_log, "Retrieved")

    result = {
        "model": args.model,
        "iters": args.iters,
        "max_tokens": args.max_tokens,
        "cold_latency": summarize(cold_latencies),
        "clear_l1_latency": summarize(clear_latencies),
        "l2_hit_latency": summarize(l2_latencies),
        "new_prefetch_completed_logs": after_prefetch - before_prefetch,
        "new_retrieved_logs": after_retrieved - before_retrieved,
        "rows": rows,
        "recent_prefetch_lines": tail_matching_lines(
            args.lmcache_log, "Prefetch request completed", 20
        ),
        "recent_retrieved_lines": tail_matching_lines(args.lmcache_log, "Retrieved", 20),
    }

    print("\n=== JSON ===")
    print(json.dumps(result, indent=2))

    print("\n=== Summary ===")
    print("metric | count | mean_ms | median_ms | min_ms | max_ms | p90_ms")
    print("--- | --- | --- | --- | --- | --- | ---")
    for name, stat in [
        ("cold", result["cold_latency"]),
        ("clear_l1", result["clear_l1_latency"]),
        ("l2_hit", result["l2_hit_latency"]),
    ]:
        print(
            f"{name} | {stat['count']} | {stat['mean_ms']:.2f} | "
            f"{stat['median_ms']:.2f} | {stat['min_ms']:.2f} | "
            f"{stat['max_ms']:.2f} | {stat['p90_ms']:.2f}"
        )

    print("\nnew_prefetch_completed_logs:", result["new_prefetch_completed_logs"])
    print("new_retrieved_logs:", result["new_retrieved_logs"])


if __name__ == "__main__":
    main()
