#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import subprocess
from pathlib import Path

# Directories to scan
EXAMPLES_DIRS = [
    "examples/offline_inference",
    "examples/online_serving",
]

# Known distributed, TPU, and Neuron examples to skip
SKIP_KEYWORDS = [
    "torchrun", "data_parallel", "disaggregated", "rlhf", "tpu", "neuron",
    "ray", "eagle", "multimodal", "speculation", "colocate", "multi_instance",
    "batch_llm_inference", "profiling_tpu", "openai_batch", "qwen2_5_omni",
    "profiling.py", "prithvi_geospatial_mae.py"
]

RESULTS = []


def is_single_gpu_example(fpath):
    fname = os.path.basename(fpath)
    # Skip if any skip keyword is in the path or filename
    return not any(kw in fpath or kw in fname for kw in SKIP_KEYWORDS)


def main():
    example_files = []
    for ex_dir in EXAMPLES_DIRS:
        for root, _, files in os.walk(ex_dir):
            for fname in files:
                if fname.endswith(".py") and is_single_gpu_example(fname):
                    fpath = os.path.join(root, fname)
                    example_files.append(fpath)

    print(f"Testing {len(example_files)} examples with VLLM_USE_V1=1...")
    summary = []
    failed = []
    for fpath in sorted(example_files):
        print(f"\nRunning: {fpath}")
        try:
            result = subprocess.run(
                ["python", fpath],
                env={
                    **os.environ, "VLLM_USE_V1": "1"
                },
                capture_output=True,
                timeout=180,
                text=True,
            )
            if result.returncode == 0:
                status = "PASS"
            else:
                status = "FAIL"
                failed.append((fpath, result.stderr.strip()
                               or result.stdout.strip()))
            print(result.stdout)
            print(result.stderr)
        except Exception as e:
            status = f"ERROR: {e}"
        summary.append((fpath, status))

    print("\nSummary:")
    print("| File | Status |\n|------|--------|")
    for fpath, status in summary:
        print(f"| {fpath} | {status} |")

    # Write failed tests to file
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "failed_single_gpu_examples.txt",
              "w",
              encoding="utf-8") as f:
        for fpath, error in failed:
            f.write(f"{fpath}\nERROR:\n{error}\n{'-'*60}\n")


if __name__ == "__main__":
    main()
