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

    const child = spawn(pythonBin, [executorScript], {
      cwd: repoRoot,
      env: { ...process.env, MPLBACKEND: "Agg" },
      timeout: timeoutMs,
    });

    let stdoutData = "";
    let stderrData = "";

    child.stdout.on("data", (chunk) => {
      stdoutData += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderrData += chunk.toString();
    });

    child.on("error", (err) => {
      resolve({
        success: false,
        stdout: "",
        stderr: stderrData,
        error: `Process execution error: ${err.message}`,
        plots: [],
        execution_duration_seconds: 0,
      });
    });

    child.on("close", (code) => {
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
    });

    child.stdin.write(code);
    child.stdin.end();
  });
}
