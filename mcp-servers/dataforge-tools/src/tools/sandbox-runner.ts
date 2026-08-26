import { spawn } from "child_process";
import * as path from "path";

export interface SandboxExecutionResult {
  success: boolean;
  stdout: string;
  stderr: string;
  error?: string | null;
  traceback?: string | null;
  plots: Array<{
    filename: string;
    filepath: string;
    absolute_path: string;
  }>;
  execution_duration_seconds: number;
}

export async function runPythonInSandbox(
  code: string,
  repoRoot: string,
  timeoutMs: number = 60000
): Promise<SandboxExecutionResult> {
  return new Promise((resolve) => {
    const pythonBin = path.join(repoRoot, ".venv", "bin", "python");
    const executorScript = path.join(repoRoot, "sandbox-env", "executor.py");

    // Clean, restricted environment variables for sandbox process
    const cleanEnv = {
      PATH: process.env.PATH || "/usr/local/bin:/usr/bin:/bin",
      VIRTUAL_ENV: path.join(repoRoot, ".venv"),
      MPLBACKEND: "Agg",
      PYTHONUNBUFFERED: "1",
    };

    const child = spawn(pythonBin, [executorScript, "--timeout", String(Math.floor(timeoutMs / 1000))], {
      cwd: repoRoot,
      env: cleanEnv,
      timeout: timeoutMs,
      detached: process.platform !== "win32", // Allows killing entire process group if needed
    });

    let stdoutData = "";
    let stderrData = "";
    let isResolved = false;

    const timer = setTimeout(() => {
      if (!isResolved) {
        isResolved = true;
        try {
          if (child.pid && process.platform !== "win32") {
            process.kill(-child.pid, "SIGKILL");
          } else {
            child.kill("SIGKILL");
          }
        } catch {
          // Process might already be dead
        }
        resolve({
          success: false,
          stdout: stdoutData,
          stderr: `TimeoutError: Sandbox execution exceeded ${timeoutMs / 1000}s limit.`,
          error: "Sandbox execution timeout",
          plots: [],
          execution_duration_seconds: timeoutMs / 1000,
        });
      }
    }, timeoutMs + 1000);

    child.stdout.on("data", (chunk) => {
      stdoutData += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderrData += chunk.toString();
    });

    child.on("error", (err) => {
      if (!isResolved) {
        isResolved = true;
        clearTimeout(timer);
        resolve({
          success: false,
          stdout: "",
          stderr: stderrData,
          error: `Process execution error: ${err.message}`,
          plots: [],
          execution_duration_seconds: 0,
        });
      }
    });

    child.on("close", (code) => {
      if (!isResolved) {
        isResolved = true;
        clearTimeout(timer);
        try {
          const parsed: SandboxExecutionResult = JSON.parse(stdoutData);
          resolve(parsed);
        } catch {
          resolve({
            success: code === 0,
            stdout: stdoutData,
            stderr: stderrData,
            error: code !== 0 ? `Process exited with code ${code}` : undefined,
            plots: [],
            execution_duration_seconds: 0,
          });
        }
      }
    });

    child.stdin.write(code);
    child.stdin.end();
  });
}
