"""Run the reproducible product discontinuation ML pipeline."""

import subprocess
import sys


def run_step(module):
    print(f"\nRunning {module}...", flush=True)
    subprocess.run(
        [sys.executable, "-m", module],
        check=True
    )


def main():
    steps = [
        "src.build_dataset",
        "src.train_logistic",
        "src.train_random_forest",
        "src.tune_logistic",
        "src.tune_random_forest",
        "src.final_evaluation",
        "src.evaluate_threshold",
        "src.monitoring",
    ]

    for step in steps:
        run_step(step)

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()