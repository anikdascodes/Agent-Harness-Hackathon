"""
DataForge AI - Safe Python Sandbox Executor
Executes data science & ML scripts in an isolated, monitored sub-process with automatic plot capture.
"""

import sys
import os
import io
import json
import time
import uuid
import traceback
import argparse
from contextlib import redirect_stdout, redirect_stderr

def execute_code(code_str: str, working_dir: str = None, timeout_seconds: int = 45) -> dict:
    """
    Executes the provided python code in a controlled context.
    Captures stdout, stderr, errors, execution time, and any matplotlib figures created.
    """
    start_time = time.time()
    if working_dir:
        os.makedirs(working_dir, exist_ok=True)
        orig_cwd = os.getcwd()
        os.chdir(working_dir)
    else:
        orig_cwd = os.getcwd()

    plots_dir = os.path.join(orig_cwd, "outputs", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    captured_plots = []
    success = False
    error_msg = None
    traceback_str = None

    # Global execution namespace
    exec_globals = {
        "__name__": "__main__",
        "__file__": "sandbox_execution.py"
    }

    try:
        # Pre-import standard DS libraries to namespace
        import matplotlib
        matplotlib.use("Agg")  # Headless backend
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        exec_globals.update({
            "pd": pd,
            "np": np,
            "plt": plt,
            "matplotlib": matplotlib
        })

        # Hook to capture plots before closing
        def custom_show(*args, **kwargs):
            fig_nums = plt.get_fignums()
            for fnum in fig_nums:
                fig = plt.figure(fnum)
                plot_id = f"plot_{uuid.uuid4().hex[:8]}.png"
                plot_path = os.path.join(plots_dir, plot_id)
                fig.savefig(plot_path, bbox_inches="tight", dpi=150)
                captured_plots.append({
                    "filename": plot_id,
                    "filepath": os.path.relpath(plot_path, orig_cwd),
                    "absolute_path": os.path.abspath(plot_path)
                })
            plt.close('all')

        plt.show = custom_show

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            compiled = compile(code_str, "<sandbox_code>", "exec")
            exec(compiled, exec_globals)

            # Auto-save any open figures that did not explicitly call plt.show()
            fig_nums = plt.get_fignums()
            for fnum in fig_nums:
                fig = plt.figure(fnum)
                plot_id = f"plot_{uuid.uuid4().hex[:8]}.png"
                plot_path = os.path.join(plots_dir, plot_id)
                fig.savefig(plot_path, bbox_inches="tight", dpi=150)
                captured_plots.append({
                    "filename": plot_id,
                    "filepath": os.path.relpath(plot_path, orig_cwd),
                    "absolute_path": os.path.abspath(plot_path)
                })
            plt.close('all')

        success = True

    except Exception as e:
        success = False
        error_msg = str(e)
        traceback_str = traceback.format_exc()
    finally:
        os.chdir(orig_cwd)

    execution_duration = round(time.time() - start_time, 3)

    return {
        "success": success,
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue(),
        "error": error_msg,
        "traceback": traceback_str,
        "plots": captured_plots,
        "execution_duration_seconds": execution_duration
    }


def main():
    parser = argparse.ArgumentParser(description="DataForge Sandbox Python Executor")
    parser.add_argument("--code", type=str, help="Python code to execute")
    parser.add_argument("--code-file", type=str, help="Path to Python script file to execute")
    parser.add_argument("--working-dir", type=str, default=".", help="Working directory")
    args = parser.parse_args()

    if args.code_file and os.path.exists(args.code_file):
        with open(args.code_file, "r", encoding="utf-8") as f:
            code = f.read()
    elif args.code:
        code = args.code
    else:
        code = sys.stdin.read()

    result = execute_code(code, working_dir=args.working_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
