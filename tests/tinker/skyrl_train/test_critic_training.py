"""Run the three stock-Tinker FSDP critic integration clients.

This is a standalone GPU test driver, not a pytest module. For each case it
starts ``skyrl.tinker.api`` as a separate command, waits for the health endpoint,
runs the corresponding standalone Tinker client, and then stops the entire
server process group before continuing.

The cases cover policy-only training on one GPU, policy and critic training on
dedicated GPUs, and fractional policy/critic sharing on one GPU while vLLM uses
the second GPU. Models are loaded from the Hugging Face cache (or downloaded)
using the hard-coded ``Qwen/Qwen3-0.6B`` model ID.

Run from the repository root with two visible GPUs:

    python tests/tinker/skyrl_train/test_critic_training.py
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


MODEL = "Qwen/Qwen3-0.6B"
PORT = 18080
BASE_URL = f"http://127.0.0.1:{PORT}"
PROJECT_ROOT = Path(__file__).parents[3]
CLIENT_DIR = Path(__file__).parent / "critic_training"

COMMON_CONFIG = {
    "trainer.placement.policy_num_gpus_per_node": 1,
    "trainer.placement.policy_num_nodes": 1,
    "trainer.placement.colocate_all": False,
    "trainer.micro_train_batch_size_per_gpu": 1,
    "trainer.micro_forward_batch_size_per_gpu": 1,
}
CRITIC_CONFIG = {
    **COMMON_CONFIG,
    "trainer.placement.critic_num_gpus_per_node": 1,
    "trainer.placement.critic_num_nodes": 1,
    "trainer.critic.model.path": MODEL,
}
SHARED_CONFIG = {
    **CRITIC_CONFIG,
    "policy_gpu_fraction": 0.55,
    "critic_gpu_fraction": 0.45,
    "generator.inference_engine.num_engines": 1,
    "generator.inference_engine.tensor_parallel_size": 1,
    "generator.inference_engine.gpu_memory_utilization": 0.70,
    "generator.inference_engine.engine_init_kwargs.max_model_len": 512,
}
CASES = (
    ("policy-only", COMMON_CONFIG, "policy_only_client.py"),
    ("dedicated-critic", CRITIC_CONFIG, "dedicated_critic_client.py"),
    ("shared-critic", SHARED_CONFIG, "shared_critic_client.py"),
)


def wait_until_ready(server: subprocess.Popen, log_path: Path) -> None:
    """Wait up to five minutes for the API health endpoint."""
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Tinker server exited early:\n{log_path.read_text()}")
        try:
            with urllib.request.urlopen(f"{BASE_URL}/api/v1/healthz", timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError):
            time.sleep(1)
    raise TimeoutError(f"Tinker server did not become ready; log: {log_path}")


def stop_server(server: subprocess.Popen) -> None:
    """Stop the API server and its engine/Ray subprocesses."""
    if server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=10)


def run_case(name: str, backend_config: dict, client_script: str, state_root: Path) -> None:
    """Start one server configuration and run its standalone Tinker client."""
    state = state_root / name
    state.mkdir()
    log_path = state / "server.log"
    server_command = [
        "uv",
        "run",
        "--isolated",
        "--extra",
        "tinker",
        "--extra",
        "fsdp",
        "-m",
        "skyrl.tinker.api",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--base-model",
        MODEL,
        "--backend",
        "fsdp",
        "--database-url",
        f"sqlite:///{state / 'tinker.db'}",
        "--checkpoints-base",
        str(state / "checkpoints"),
        "--backend-config",
        json.dumps(backend_config),
    ]
    client_command = [
        "uv",
        "run",
        "--isolated",
        "--extra",
        "tinker",
        "python",
        str(CLIENT_DIR / client_script),
    ]

    print(f"\n=== {name} ===", flush=True)
    print(f"$ {shlex.join(server_command)}", flush=True)
    with log_path.open("w") as log:
        server = subprocess.Popen(
            server_command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_until_ready(server, log_path)
            print(f"$ {shlex.join(client_command)}", flush=True)
            subprocess.run(client_command, cwd=PROJECT_ROOT, check=True)
        except BaseException:
            log.flush()
            print(f"\n=== {name} server log ===\n{log_path.read_text()}", file=sys.stderr)
            raise
        finally:
            stop_server(server)


def main() -> None:
    """Run all configurations sequentially with isolated server state."""
    with tempfile.TemporaryDirectory(prefix="skyrl-critic-training-") as state_root:
        for case in CASES:
            run_case(*case, Path(state_root))
    print("\nAll critic-training clients passed", flush=True)


if __name__ == "__main__":
    main()
