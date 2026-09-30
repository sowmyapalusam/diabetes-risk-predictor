"""Simple inference benchmark for the diabetes risk predictor."""
from __future__ import annotations

from model import benchmark_inference, train_models


def main() -> None:
    """Train the model once and report inference latency."""
    random_forest, _, _, _ = train_models()
    result = benchmark_inference(random_forest, batch_size=128)
    print(f"Batch size: {int(result['batch_size'])}")
    print(f"Inference time: {result['elapsed_ms']:.2f} ms")
    print(f"Throughput: {result['rows_per_second']:.2f} rows/s")


if __name__ == "__main__":
    main()
