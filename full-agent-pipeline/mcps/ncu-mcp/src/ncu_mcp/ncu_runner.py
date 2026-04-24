"""Async subprocess wrapper for ncu CLI."""

import asyncio
import os
from dataclasses import dataclass


NCU_PATH = os.environ.get("NCU_PATH", "ncu")
REPORTS_DIR = os.path.join(os.getcwd(), "reports")


@dataclass
class NCUResult:
    stdout: str
    stderr: str
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def error_message(self) -> str:
        if self.ok:
            return ""
        msg = self.stderr.strip() or self.stdout.strip()
        return msg or f"ncu exited with code {self.returncode}"


async def run_ncu(args: list[str], timeout: int = 300, env: dict | None = None) -> NCUResult:
    """Run ncu with the given arguments and return structured result."""
    cmd = [NCU_PATH] + args
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=run_env,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        return NCUResult(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            returncode=proc.returncode,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return NCUResult(stdout="", stderr=f"ncu timed out after {timeout}s", returncode=-1)
    except FileNotFoundError:
        return NCUResult(stdout="", stderr=f"ncu not found at '{NCU_PATH}'. Set NCU_PATH env var.", returncode=-1)
    except Exception as e:
        return NCUResult(stdout="", stderr=f"Failed to run ncu: {e}", returncode=-1)


def report_path(name: str) -> str:
    """Return full path for a report name (without extension)."""
    return os.path.join(REPORTS_DIR, f"{name}.ncu-rep")
