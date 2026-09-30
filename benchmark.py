"""Performance benchmark for the diabetes risk predictor."""

from __future__ import annotations

import time
import tracemalloc

from model import benchmark_inference, train_models


def run_benchmark(batch_size: int, model) -> dict[str, float]:
    """Measure inference latency, throughput, and peak memory."""

    tracemalloc.start()

    start = time.perf_counter()
    result = benchmark_inference(model, batch_size=batch_size)
    elapsed = time.perf_counter() - start

    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "batch_size": float(batch_size),
        "latency_ms": float(result["elapsed_ms"]),
        "throughput_rows_per_second": float(result["rows_per_second"]),
        "peak_memory_mb": peak_memory / (1024 * 1024),
        "wall_time_ms": elapsed * 1000,
    }


def main() -> None:
    """Run benchmarks across multiple inference batch sizes."""

    random_forest, _, _, _ = train_models()

    print("Diabetes Risk Predictor - Performance Benchmark")
    print("=" * 55)

    for batch_size in (32, 128, 512):
        result = run_benchmark(batch_size, random_forest)

        print(
            f"Batch {int(result['batch_size']):>3} | "
            f"Latency: {result['latency_ms']:.2f} ms | "
            f"Throughput: {result['throughput_rows_per_second']:.2f} rows/s | "
            f"Peak memory: {result['peak_memory_mb']:.2f} MB"
        )


if __name__ == "__main__":
    main()
