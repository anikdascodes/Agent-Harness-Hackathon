"""
DataForge AI - Safe Python Sandbox Executor
Hardened execution environment with AST-based safety checks, true process-level timeout enforcement,
and automatic Matplotlib/Seaborn plot capture.
"""

import sys
import os
import io
import ast
import json
import time
import uuid
import traceback
import argparse
import multiprocessing
from contextlib import redirect_stdout, redirect_stderr

# Restricted modules and system functions for safe execution
BLOCKED_MODULES = {
    "subprocess", "pty", "socket", "http", "urllib", "requests",
    "shutil", "telnetlib", "ftplib", "smtplib", "webbrowser",
    "ctypes", "winreg"
}

BLOCKED_CALLS = {
    "system", "popen", "execv", "execve", "spawn", "fork", "kill"
}

def validate_code_safety(code_str: str) -> tuple[bool, str | None]:
    """
    Performs static AST validation to block dangerous system-level calls
    and unauthorized process/network modules before execution.
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"Syntax Error: {e.msg} at line {e.lineno}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split('.')[0]
                if root_module in BLOCKED_MODULES:
                    return False, f"Security Violation: Import of module '{root_module}' is restricted in sandbox."
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_module = node.module.split('.')[0]
                if root_module in BLOCKED_MODULES:
                    return False, f"Security Violation: Import from module '{root_module}' is restricted in sandbox."
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in BLOCKED_CALLS:
                    return False, f"Security Violation: Call to dangerous function '{node.func.attr}()' is restricted."

    return True, None


def _sandbox_process_target(code_str: str, orig_cwd: str, plots_dir: str, result_pipe):
    """Target function executed in a dedicated, killable child process."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    captured_plots = []
    success = False
    error_msg = None
    traceback_str = None

    exec_globals = {
        "__name__": "__main__",
        "__file__": "sandbox_execution.py"
    }

    try:
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

    result_pipe.send({
        "success": success,
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue(),
        "error": error_msg,
        "traceback": traceback_str,
        "plots": captured_plots
    })


def execute_code(code_str: str, working_dir: str = None, timeout_seconds: int = 45) -> dict:
    """
    Executes the provided python code in a controlled, monitored sandbox.
    Validates AST safety, enforces hard process-level timeout (terminating runaway execution),
    and captures all visual plots.
    """
    start_time = time.time()
    
    # 1. AST Safety Validation
    is_safe, security_err = validate_code_safety(code_str)
    if not is_safe:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"SecurityError: {security_err}",
            "error": security_err,
            "traceback": None,
            "plots": [],
            "execution_duration_seconds": 0.0
        }

    orig_cwd = os.getcwd()
    if working_dir:
        os.makedirs(working_dir, exist_ok=True)
        os.chdir(working_dir)

    plots_dir = os.path.join(orig_cwd, "outputs", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    proc = multiprocessing.Process(
        target=_sandbox_process_target,
        args=(code_str, orig_cwd, plots_dir, child_conn)
    )

    proc.start()
    proc.join(timeout=timeout_seconds)

    if proc.is_alive():
        # Hard process-level termination of runaway/hanging code
        proc.kill()
        proc.join(timeout=1.0)
        os.chdir(orig_cwd)
        return {
            "success": False,
            "stdout": "",
            "stderr": f"TimeoutError: Sandbox execution exceeded {timeout_seconds}s limit.",
            "error": f"TimeoutError: Sandbox execution exceeded {timeout_seconds}s limit.",
            "traceback": None,
            "plots": [],
            "execution_duration_seconds": round(time.time() - start_time, 3)
        }

    os.chdir(orig_cwd)

    if parent_conn.poll():
        data = parent_conn.recv()
        data["execution_duration_seconds"] = round(time.time() - start_time, 3)
        return data
    else:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Execution process exited abnormally without returning output.",
            "error": "Process abnormally terminated",
            "traceback": None,
            "plots": [],
            "execution_duration_seconds": round(time.time() - start_time, 3)
        }


def main():
    parser = argparse.ArgumentParser(description="DataForge Sandbox Python Executor")
    parser.add_argument("--code", type=str, help="Python code to execute")
    parser.add_argument("--code-file", type=str, help="Path to Python script file to execute")
    parser.add_argument("--working-dir", type=str, default=".", help="Working directory")
    parser.add_argument("--timeout", type=int, default=45, help="Execution timeout in seconds")
    args = parser.parse_args()

    if args.code_file and os.path.exists(args.code_file):
        with open(args.code_file, "r", encoding="utf-8") as f:
            code = f.read()
    elif args.code:
        code = args.code
    else:
        code = sys.stdin.read()

    result = execute_code(code, working_dir=args.working_dir, timeout_seconds=args.timeout)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
