import asyncio
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import asyncpg
import httpx

READY_TIMEOUT = 30.0
STOP_TIMEOUT = 10.0


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def base_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith("VECTORIST_")}


def server_env(backend: str, tmp_path: Path) -> dict[str, str]:
    env = base_env()
    env["VECTORIST_CACHE_PATH"] = str(tmp_path / "cache.db")
    env["VECTORIST_VEC_PATH"] = str(tmp_path / "vec.db")
    env["VECTORIST_EMBEDDER"] = "fake"
    if backend == "pgvector":
        env["VECTORIST_VECTOR_BACKEND"] = "pgvector"
        env["VECTORIST_DATABASE_URL"] = os.environ["VECTORIST_PG_URL"]
    return env


def truncate_vector_chunks(dsn: str) -> None:
    async def run() -> None:
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute("TRUNCATE vector_chunks")
        finally:
            await conn.close()

    asyncio.run(run())


class Process:
    def __init__(self, proc: subprocess.Popen, stderr_file) -> None:
        self.proc = proc
        self._stderr_file = stderr_file

    def stderr(self) -> str:
        self._stderr_file.flush()
        return Path(self._stderr_file.name).read_text()

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.send_signal(signal.SIGTERM)
                try:
                    self.proc.wait(timeout=STOP_TIMEOUT)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        finally:
            self._stderr_file.close()
            Path(self._stderr_file.name).unlink(missing_ok=True)


def start_server(env: dict[str, str], port: int) -> Process:
    stderr_file = tempfile.NamedTemporaryFile("w+", suffix=".stderr", delete=False)
    try:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "vectorist.api.app:create_app",
                "--factory", "--host", "127.0.0.1", "--port", str(port),
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
        )
    except BaseException:
        stderr_file.close()
        Path(stderr_file.name).unlink(missing_ok=True)
        raise
    return Process(proc, stderr_file)


def wait_ready(process: Process, port: int) -> None:
    deadline = time.monotonic() + READY_TIMEOUT
    url = f"http://127.0.0.1:{port}/v1/health"
    while True:
        code = process.proc.poll()
        if code is not None:
            raise RuntimeError(f"server exited with {code}:\n{process.stderr()}")
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                return
        except httpx.TransportError:
            pass
        if time.monotonic() > deadline:
            raise RuntimeError(f"server not ready in {READY_TIMEOUT}s:\n{process.stderr()}")
        time.sleep(0.1)


def wait_exit(process: Process, timeout: float) -> int:
    try:
        return process.proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"server still running after {timeout}s:\n{process.stderr()}")


class LiveServer:
    def __init__(self, env: dict[str, str]) -> None:
        self.env = env
        self.port = free_port()
        self._process: Process | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self._process = start_server(self.env, self.port)
        try:
            wait_ready(self._process, self.port)
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        if self._process is not None:
            process, self._process = self._process, None
            process.close()
