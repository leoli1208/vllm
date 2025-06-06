#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys
from pathlib import Path

def run_example(script_path):
    print(f"\nRunning {script_path}...")
    try:
        # Set V1 environment variable for this run
        env = os.environ.copy()
        env["VLLM_USE_V1"] = "1"
        
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, 
                              text=True,
                              env=env)
        if result.returncode == 0:
            print("✅ Success!")
        else:
            print(f"❌ Failed with error:\n{result.stderr}")
    except Exception as e:
        print(f"❌ Error running {script_path}: {str(e)}")

def main():
    # Get the examples directory path
    examples_dir = Path("examples/offline_inference/basic")
    
    if not examples_dir.exists():
        print(f"Error: {examples_dir} does not exist")
        return

    print("=== Running Basic Examples with VLLM_USE_V1=1 ===")
    
    # Run all Python files in the basic examples directory
    for script in examples_dir.glob("*.py"):
        if script.name != "__init__.py":
            run_example(str(script))

if __name__ == "__main__":
    main()
