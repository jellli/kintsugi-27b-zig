#!/usr/bin/env python3
"""End-to-end head-to-head benchmark: llama-server (Kintsugi-27B tuned) vs kintsugi-zig.

Runs both engines sequentially on the same GPU (zero contention) with the exact
same Kintsugi-27B + DFlash2 GGUF weights, measuring:
  1. Cold startup time, VRAM footprint (idle & peak), and VRAM Q4_0 KV capacity
  2. Pure M=1 (non-speculative) baseline decode throughput
  3. Diverse short-context speculative decode workloads (greedy + temp=0.7) & parity
  4. Chat API streaming (thinking OFF & thinking ON)
  5. Long-context prefill scaling, decode speed & 3-needle retrieval (2K..76K)
  6. Multi-turn ~10K system-prompt prefix cache reuse (Turn-1 cold vs Turn-2 cached)

Outputs a side-by-side Markdown comparison table and raw JSON results.
Zero third-party Python dependencies (stdlib only).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import statistics
import string
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZIG_BIN = REPO_ROOT / "zig-out" / "bin" / "kintsugi-zig"
DEFAULT_LLAMA_ROOT = Path("/home/jili/personal/kintsugi-27b/Kintsugi-27B")
DEFAULT_MODEL = DEFAULT_LLAMA_ROOT / "models" / "Kintsugi-27B-12G.gguf"
DEFAULT_DRAFTER = DEFAULT_LLAMA_ROOT / "models" / "dflash-Qwen3.8-27B-Q4_0.gguf"

# Empirically ~1.052 tokens/word on Qwen3.8 tokenizer
DOC_WORDS = (
    "river stone paper cloud engine garden silver market winter orange signal "
    "harbor violet candle forest mirror thunder copper meadow lantern"
).split()

SHORT_CASES: list[dict[str, Any]] = [
    {
        "id": "quicksort_en",
        "label": "Quicksort (EN Code, T=0)",
        "prompt": (
            "Write a concise Python implementation of quicksort and explain its "
            "time complexity in detail."
        ),
        "max_tokens": 256,
        "temperature": 0.0,
        "parity": True,
    },
    {
        "id": "code_edit",
        "label": "Code Refactor (PLD, T=0)",
        "prompt": (
            "Refactor the following Python class to add type annotations and a "
            "`clear()` method while preserving every existing method name, "
            "docstring, and behavior verbatim:\n\n"
            "```python\n"
            "class RingBuffer:\n"
            "    \"\"\"Fixed-capacity circular buffer for numeric telemetry.\"\"\"\n"
            "    def __init__(self, capacity):\n"
            "        if capacity <= 0:\n"
            "            raise ValueError('capacity must be positive')\n"
            "        self._capacity = capacity\n"
            "        self._data = [0.0] * capacity\n"
            "        self._head = 0\n"
            "        self._size = 0\n\n"
            "    def push(self, value):\n"
            "        \"\"\"Insert a value, overwriting the oldest element when full.\"\"\"\n"
            "        idx = (self._head + self._size) % self._capacity\n"
            "        if self._size == self._capacity:\n"
            "            self._head = (self._head + 1) % self._capacity\n"
            "        else:\n"
            "            self._size += 1\n"
            "        self._data[idx] = value\n\n"
            "    def values(self):\n"
            "        \"\"\"Return stored values from oldest to newest.\"\"\"\n"
            "        return [\n"
            "            self._data[(self._head + i) % self._capacity]\n"
            "            for i in range(self._size)\n"
            "        ]\n"
            "```\n\n"
            "Refactored Python code:\n```python\n"
        ),
        "max_tokens": 256,
        "temperature": 0.0,
        "parity": True,
    },
    {
        "id": "qa_en",
        "label": "Factual QA (EN, T=0)",
        "prompt": (
            "Explain how TCP congestion control (AIMD, slow start, fast retransmit, "
            "and fast recovery) prevents network collapse. Be precise and structured."
        ),
        "max_tokens": 256,
        "temperature": 0.0,
        "parity": True,
    },
    {
        "id": "reason_zh",
        "label": "Math Reasoning (ZH, T=0)",
        "prompt": (
            "请逐步推导并证明：对于任意正整数 n，1^3 + 2^3 + ... + n^3 = (n(n+1)/2)^2。"
            "要求先用数学归纳法给出严谨证明，再举例验证 n=4。"
        ),
        "max_tokens": 256,
        "temperature": 0.0,
        "parity": True,
    },
    {
        "id": "creative_zh",
        "label": "Creative Prose (ZH Floor, T=0)",
        "prompt": (
            "写一段约300字的雨夜江南古镇散文，注重细节描写、光影变化与人物心境，不要使用列表："
        ),
        "max_tokens": 256,
        "temperature": 0.0,
        "parity": True,
    },
    {
        "id": "quicksort_t07",
        "label": "Quicksort (EN, T=0.7)",
        "prompt": (
            "Write a concise Python implementation of quicksort and explain its "
            "time complexity in detail."
        ),
        "max_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "seed": 42,
        "parity": False,
    },
    {
        "id": "creative_zh_t07",
        "label": "Creative Prose (ZH, T=0.7)",
        "prompt": (
            "写一段约300字的雨夜江南古镇散文，注重细节描写、光影变化与人物心境，不要使用列表："
        ),
        "max_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "seed": 42,
        "parity": False,
    },
]


def query_gpu_info() -> dict[str, Any]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    parts = [p.strip() for p in out.split(",")]
    return {
        "name": parts[0],
        "memory_total_mib": int(parts[1]),
        "memory_used_mib": int(parts[2]),
        "memory_free_mib": int(parts[3]),
        "driver_version": parts[4],
    }


def query_proc_vram(pid: int) -> tuple[int, int]:
    """Return (proc_used_mib, gpu_free_mib)."""
    free_out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    free_mib = int(free_out.splitlines()[0].strip())
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    )
    proc_mib = 0
    pid_prefix = f"{pid},"
    for line in apps.splitlines():
        s = line.strip()
        if s.startswith(pid_prefix):
            try:
                proc_mib = int(s.split(",")[1].strip())
            except ValueError:
                pass
    return proc_mib, free_mib


class VramMonitor:
    def __init__(self, pid: int, interval: float = 0.5) -> None:
        self.pid = pid
        self.interval = interval
        self.peak_proc_mib = 0
        self.min_free_mib = 10**9
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def _sample(self) -> None:
        try:
            proc_mib, free_mib = query_proc_vram(self.pid)
            if proc_mib > self.peak_proc_mib:
                self.peak_proc_mib = proc_mib
            if free_mib < self.min_free_mib:
                self.min_free_mib = free_mib
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def stop(self) -> dict[str, int]:
        self._sample()
        self._stop.set()
        self._thread.join(timeout=3.0)
        return {
            "peak_proc_mib": self.peak_proc_mib,
            "min_free_mib": self.min_free_mib if self.min_free_mib < 10**9 else 0,
        }


class ManagedServer:
    def __init__(
        self,
        name: str,
        cmd: list[str],
        env: dict[str, str],
        port: int,
        log_path: Path,
    ) -> None:
        self.name = name
        self.cmd = cmd
        self.env = env
        self.port = port
        self.log_path = log_path
        self.proc: subprocess.Popen[bytes] | None = None
        self.log_file: Any = None
        self.startup_s: float = 0.0
        self.health_json: dict[str, Any] = {}
        self.idle_vram_mib: int = 0

    def start(self, timeout_s: float = 180.0) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, "wb")
        t0 = time.monotonic()
        self.proc = subprocess.Popen(
            self.cmd,
            env=self.env,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        url = f"http://127.0.0.1:{self.port}/health"
        deadline = t0 + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.log_file.flush()
                tail = self.log_path.read_text(errors="replace")[-1500:]
                raise RuntimeError(
                    f"{self.name} exited early (code={self.proc.returncode}):\n{tail}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2.0) as resp:
                    if resp.status == 200:
                        raw = resp.read().decode("utf-8", "replace")
                        self.startup_s = time.monotonic() - t0
                        try:
                            self.health_json = json.loads(raw)
                        except Exception:
                            self.health_json = {"raw": raw}
                        proc_mib, _ = query_proc_vram(self.proc.pid)
                        self.idle_vram_mib = proc_mib
                        return
            except Exception:
                time.sleep(0.1)
        raise TimeoutError(f"{self.name} did not become healthy within {timeout_s}s")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=5.0)
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        # Give CUDA driver a moment to release VRAM cleanly before next process
        time.sleep(1.0)


def build_llama_spec(
    llama_root: Path,
    model: Path,
    drafter: Path,
    port: int,
    speculative: bool = True,
) -> tuple[list[str], dict[str, str]]:
    bin_dir = llama_root / "bin"
    env = os.environ.copy()
    ld = str(bin_dir)
    if env.get("LD_LIBRARY_PATH"):
        ld = f"{ld}:{env['LD_LIBRARY_PATH']}"
    env.update(
        {
            "LD_LIBRARY_PATH": ld,
            "GGML_CUDA_BATCH_INVARIANT": "1",
            "GGML_CUDA_DEDICATED_CEILING_MIB": "11914",
            "GGML_CUDA_POOL_RESERVE_MIB": "96",
            "GGML_CUDA_SHARED_POOL": "1",
            "GGML_ESCHA_I8_MIN_ROWS": "256",
            "GGML_ESCHA_PREFILL": "i8",
            "LLAMA_ARG_CHAT_TEMPLATE_KWARGS": '{"reasoning_effort":"medium"}',
            "LLAMA_ARG_CHECKPOINT_EVERY_NT": "8192",
            "LLAMA_DFLASH_DRAFT_UBATCH": "64",
            "LLAMA_HEAD2": "64",
            "LLAMA_HEAD2_TYPE": "q4_0",
            "LLAMA_MTP_DRAFT_UBATCH": "256",
            "LLAMA_RS_REPLAY": "1",
        }
    )
    cmd = [
        str(bin_dir / "llama-server"),
        "-m", str(model),
        "-ngl", "99",
        "-c", "262144",
        "-fa", "on",
        "-np", "1",
        "-ub", "1024",
        "--fit", "off",
        "--jinja",
        "-ctk", "q4_0",
        "-ctv", "q4_0",
        "--kv-vram-cells", "69632",
        "--kq-mask-packed",
        "--cache-ram", "6144",
        "--backend-sampling",
        "--reasoning-budget", "20480",
        "-n", "24576",
        "--reasoning-budget-message", "Now produce the complete answer.",
        "--reasoning-effort-allow", "medium",
        "--reasoning-effort-fallback", "medium",
        "--reasoning-max-tokens-floor", "24576",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--chat-template-file", str(llama_root / "templates" / "kintsugi.jinja"),
        "--alias", "kintsugi-27b",
    ]
    if speculative:
        cmd.extend(
            [
                "--spec-type", "draft-dflash",
                "-md", str(drafter),
                "--spec-draft-n-max", "7",
                "-ctkd", "q4_0",
                "-ctvd", "q4_0",
            ]
        )
    else:
        cmd.extend(["--spec-type", "none"])
    return cmd, env


def build_zig_spec(
    zig_bin: Path,
    model: Path,
    drafter: Path,
    port: int,
    speculative: bool = True,
) -> tuple[list[str], dict[str, str]]:
    cmd = [
        str(zig_bin),
        "serve",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--model", str(model),
        "--draft", str(drafter),
    ]
    if not speculative:
        cmd.append("--no-spec")
    return cmd, os.environ.copy()


def sse_request(
    port: int,
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float = 900.0,
) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}{endpoint}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    t_first: float | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    finish_reason = ""

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            if not data:
                continue
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            if ev.get("timings"):
                timings = ev["timings"]
            for ch in ev.get("choices") or []:
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]
                if endpoint.endswith("/chat/completions"):
                    delta = ch.get("delta") or {}
                    c = delta.get("content") or ""
                    rc = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if c or rc:
                        if t_first is None:
                            t_first = time.monotonic()
                    if c:
                        content_parts.append(c)
                    if rc:
                        reasoning_parts.append(rc)
                else:
                    txt = ch.get("text") or ""
                    if txt:
                        if t_first is None:
                            t_first = time.monotonic()
                        content_parts.append(txt)

    t_end = time.monotonic()
    if t_first is None:
        t_first = t_end

    pt = int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0)
    ct = int(usage.get("completion_tokens") or timings.get("predicted_n") or 0)
    ttft_ms = (t_first - t0) * 1000.0
    total_s = t_end - t0
    gen_s = t_end - t_first
    client_decode_tps = (ct - 1) / gen_s if (ct > 1 and gen_s > 0) else 0.0
    client_prefill_tps = pt / (ttft_ms / 1000.0) if (pt > 0 and ttft_ms > 0) else 0.0

    srv_prefill_tps = float(timings.get("prompt_per_second") or 0.0)
    srv_decode_tps = float(timings.get("predicted_per_second") or 0.0)
    draft_n = int(timings.get("draft_n") or 0)
    draft_acc = int(timings.get("draft_n_accepted") or 0)
    accept_pct = (100.0 * draft_acc / draft_n) if draft_n > 0 else 0.0

    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "ttft_ms": round(ttft_ms, 2),
        "total_s": round(total_s, 3),
        "client_prefill_tps": round(client_prefill_tps, 1),
        "client_decode_tps": round(client_decode_tps, 2),
        "srv_prefill_tps": round(srv_prefill_tps, 1),
        "srv_decode_tps": round(srv_decode_tps, 2),
        "draft_n": draft_n,
        "draft_n_accepted": draft_acc,
        "accept_pct": round(accept_pct, 1),
        "finish_reason": finish_reason,
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
    }


def make_needle_doc(target_tokens: int, seed: int, nonce: str) -> tuple[str, list[str]]:
    rng = random.Random(seed)
    n_words = int(target_tokens / 1.052)
    words = [rng.choice(DOC_WORDS) for _ in range(n_words)]
    codes = [
        "".join(rng.choices(string.ascii_uppercase + string.digits, k=8))
        for _ in range(3)
    ]
    for i, code in enumerate(codes):
        pos = int(n_words * (i + 1) / 4)
        words.insert(pos, f". The secret code number {i + 1} is {code}. ")
    body = f"[doc {nonce}-{seed}] " + " ".join(words)
    question = (
        "\n\nList every secret code number and its value that appears in the "
        "document above, then briefly explain why indexing helps long-document search."
    )
    return body + question, codes


def make_system_doc(target_tokens: int, seed: int, nonce: str) -> tuple[str, list[str]]:
    rng = random.Random(seed)
    n_words = int(target_tokens / 1.052)
    words = [rng.choice(DOC_WORDS) for _ in range(n_words)]
    codes = [
        "".join(rng.choices(string.ascii_uppercase + string.digits, k=8))
        for _ in range(3)
    ]
    for i, code in enumerate(codes):
        pos = int(n_words * (i + 1) / 4)
        words.insert(pos, f". The secret code number {i + 1} is {code}. ")
    doc = (
        f"[ref {nonce}-{seed}] Reference Archive:\n"
        + " ".join(words)
        + "\nAnswer questions strictly using the reference archive above."
    )
    return doc, codes


def run_engine_suite(
    engine: str,
    args: argparse.Namespace,
    ctx_lengths: list[int],
    log_dir: Path,
) -> dict[str, Any]:
    assert engine in ("llama", "zig")
    model_name = "kintsugi-27b" if engine == "llama" else "Kintsugi-27B"
    res: dict[str, Any] = {"engine": engine}

    # --- 1. Pure M=1 (No-Spec) Baseline ---
    if not args.skip_m1:
        print(f"\n[{engine}] === Phase 1: Pure M=1 (No-Spec) Baseline ===", flush=True)
        if engine == "llama":
            cmd, env = build_llama_spec(args.llama_root, args.model, args.drafter, args.port, speculative=False)
        else:
            cmd, env = build_zig_spec(args.zig_bin, args.model, args.drafter, args.port, speculative=False)
        srv = ManagedServer(f"{engine}-m1", cmd, env, args.port, log_dir / f"{engine}_m1.log")
        try:
            srv.start()
            # Warmup
            sse_request(
                args.port,
                "/v1/completions",
                {
                    "model": model_name,
                    "prompt": "Hello world",
                    "max_tokens": 16,
                    "temperature": 0.0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "cache_prompt": False,
                },
            )
            m1_runs = []
            for r in range(args.runs):
                r_out = sse_request(
                    args.port,
                    "/v1/completions",
                    {
                        "model": model_name,
                        "prompt": (
                            f"[m1-{r}] Write a concise Python implementation of quicksort "
                            "and explain its time complexity in detail."
                        ),
                        "max_tokens": 128,
                        "temperature": 0.0,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "cache_prompt": False,
                    },
                )
                m1_runs.append(r_out)
                print(
                    f"  [{engine}] M=1 run {r+1}/{args.runs}: "
                    f"client={r_out['client_decode_tps']} tok/s, "
                    f"srv={r_out['srv_decode_tps']} tok/s, "
                    f"ttft={r_out['ttft_ms']} ms (ct={r_out['completion_tokens']})",
                    flush=True,
                )
            res["m1_baseline"] = {
                "startup_s": round(srv.startup_s, 2),
                "client_decode_tps": round(statistics.median(x["client_decode_tps"] for x in m1_runs), 2),
                "srv_decode_tps": round(statistics.median(x["srv_decode_tps"] for x in m1_runs), 2),
                "ttft_ms": round(statistics.median(x["ttft_ms"] for x in m1_runs), 2),
                "runs": [{k: v for k, v in x.items() if k not in ("content", "reasoning")} for x in m1_runs],
            }
        finally:
            srv.stop()

    # --- 2. Full Speculative Server (DFlash2 ON) ---
    print(f"\n[{engine}] === Phase 2: Full Speculative (DFlash2) Suite ===", flush=True)
    if engine == "llama":
        cmd, env = build_llama_spec(args.llama_root, args.model, args.drafter, args.port, speculative=True)
    else:
        cmd, env = build_zig_spec(args.zig_bin, args.model, args.drafter, args.port, speculative=True)

    srv = ManagedServer(f"{engine}-spec", cmd, env, args.port, log_dir / f"{engine}_spec.log")
    vram_mon: VramMonitor | None = None
    try:
        srv.start()
        assert srv.proc is not None
        vram_mon = VramMonitor(srv.proc.pid, interval=0.5)
        vram_mon.start()

        kv_cells = (
            int(srv.health_json.get("kv_cells", 0))
            if engine == "zig"
            else 69632
        )
        res["system"] = {
            "startup_s": round(srv.startup_s, 2),
            "idle_vram_mib": srv.idle_vram_mib,
            "kv_vram_cells": kv_cells,
        }
        print(
            f"  [{engine}] Ready in {srv.startup_s:.2f}s | "
            f"idle VRAM={srv.idle_vram_mib} MiB | kv_cells={kv_cells}",
            flush=True,
        )

        # Warmup (CUDA graph capture / caches)
        sse_request(
            args.port,
            "/v1/completions",
            {
                "model": model_name,
                "prompt": "Warmup prompt: the capital of France is",
                "max_tokens": 32,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
                "cache_prompt": False,
            },
        )

        # Section B: Short-context diverse workloads (round-robin across runs so every run is 100% cold)
        short_runs: dict[str, list[dict[str, Any]]] = {c["id"]: [] for c in SHORT_CASES}
        for r in range(args.runs):
            for c in SHORT_CASES:
                payload: dict[str, Any] = {
                    "model": model_name,
                    "prompt": c["prompt"],
                    "max_tokens": c["max_tokens"],
                    "temperature": c["temperature"],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "cache_prompt": False,
                }
                for opt in ("top_p", "top_k", "seed"):
                    if opt in c:
                        payload[opt] = c[opt]
                out = sse_request(args.port, "/v1/completions", payload)
                short_runs[c["id"]].append(out)
                print(
                    f"  [{engine}] {c['id']:16s} run {r+1}/{args.runs}: "
                    f"client={out['client_decode_tps']:6.2f} tok/s | "
                    f"srv={out['srv_decode_tps']:6.2f} tok/s | "
                    f"ttft={out['ttft_ms']:6.1f} ms | "
                    f"ct={out['completion_tokens']} | acc={out['accept_pct']}%",
                    flush=True,
                )

        res["short"] = {}
        for c in SHORT_CASES:
            runs_list = short_runs[c["id"]]
            res["short"][c["id"]] = {
                "label": c["label"],
                "prompt_tokens": runs_list[0]["prompt_tokens"],
                "completion_tokens": runs_list[0]["completion_tokens"],
                "client_decode_tps": round(statistics.median(x["client_decode_tps"] for x in runs_list), 2),
                "srv_decode_tps": round(statistics.median(x["srv_decode_tps"] for x in runs_list), 2),
                "ttft_ms": round(statistics.median(x["ttft_ms"] for x in runs_list), 2),
                "accept_pct": round(statistics.median(x["accept_pct"] for x in runs_list), 1),
                "sample_text": runs_list[0]["content"] if c.get("parity") else runs_list[0]["content"][:120],
            }

        # Section C: Chat API (Thinking OFF & Thinking ON)
        print(f"\n[{engine}] === Phase 3: Chat API (Thinking OFF / ON) ===", flush=True)
        chat_direct = sse_request(
            args.port,
            "/v1/chat/completions",
            {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "用中文简明扼要地解释什么是投机解码（Speculative Decoding），"
                            "以及为什么它能做到数学上 100% 无损。"
                        ),
                    }
                ],
                "max_tokens": 256,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
                "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        print(
            f"  [{engine}] chat_direct: client={chat_direct['client_decode_tps']} tok/s | "
            f"ttft={chat_direct['ttft_ms']} ms | ct={chat_direct['completion_tokens']}",
            flush=True,
        )

        chat_think = sse_request(
            args.port,
            "/v1/chat/completions",
            {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 17 * 29 + 143? Show brief reasoning and give the final integer.",
                    }
                ],
                "max_tokens": 512,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
                "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"},
            },
        )
        think_ok = bool(chat_think["reasoning"].strip()) and ("636" in chat_think["content"])
        print(
            f"  [{engine}] chat_think:  client={chat_think['client_decode_tps']} tok/s | "
            f"ttft={chat_think['ttft_ms']} ms | ct={chat_think['completion_tokens']} | "
            f"ok={think_ok} (reasoning_chars={len(chat_think['reasoning'])})",
            flush=True,
        )
        res["chat"] = {
            "chat_direct": {
                "prompt_tokens": chat_direct["prompt_tokens"],
                "completion_tokens": chat_direct["completion_tokens"],
                "ttft_ms": chat_direct["ttft_ms"],
                "client_decode_tps": chat_direct["client_decode_tps"],
                "srv_decode_tps": chat_direct["srv_decode_tps"],
                "finish_reason": chat_direct["finish_reason"],
            },
            "chat_think": {
                "prompt_tokens": chat_think["prompt_tokens"],
                "completion_tokens": chat_think["completion_tokens"],
                "ttft_ms": chat_think["ttft_ms"],
                "client_decode_tps": chat_think["client_decode_tps"],
                "srv_decode_tps": chat_think["srv_decode_tps"],
                "finish_reason": chat_think["finish_reason"],
                "correct_636": think_ok,
                "reasoning_chars": len(chat_think["reasoning"]),
            },
        }

        # Section D: Multi-turn ~10K system-prompt prefix cache reuse
        if not args.skip_long:
            print(f"\n[{engine}] === Phase 4: Multi-Turn ~10K Prefix Cache Reuse ===", flush=True)
            sys_doc, sys_codes = make_system_doc(10240, seed=10240, nonce=f"{engine}-mt")
            t1 = sse_request(
                args.port,
                "/v1/chat/completions",
                {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": sys_doc},
                        {"role": "user", "content": "What is secret code number 1? Reply with the code only."},
                    ],
                    "max_tokens": 32,
                    "temperature": 0.0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "cache_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            t2 = sse_request(
                args.port,
                "/v1/chat/completions",
                {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": sys_doc},
                        {"role": "user", "content": "What is secret code number 2? Reply with the code only."},
                    ],
                    "max_tokens": 32,
                    "temperature": 0.0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "cache_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            speedup = (t1["ttft_ms"] / t2["ttft_ms"]) if t2["ttft_ms"] > 0 else 0.0
            res["prefix_cache_10k"] = {
                "prompt_tokens": t1["prompt_tokens"],
                "turn1_ttft_ms": t1["ttft_ms"],
                "turn2_ttft_ms": t2["ttft_ms"],
                "speedup_x": round(speedup, 1),
                "turn1_hit": sys_codes[0] in t1["content"],
                "turn2_hit": sys_codes[1] in t2["content"],
            }
            print(
                f"  [{engine}] prefix_10k ({t1['prompt_tokens']} tok): "
                f"T1_ttft={t1['ttft_ms']:.1f} ms -> T2_ttft={t2['ttft_ms']:.1f} ms "
                f"({speedup:.1f}x) | hits={sys_codes[0] in t1['content']}/{sys_codes[1] in t2['content']}",
                flush=True,
            )

            # Section E: Long-context prefill scaling + decode + 3-needle retrieval
            print(f"\n[{engine}] === Phase 5: Long-Context Prefill, Decode & 3-Needle Recall ===", flush=True)
            res["long_ctx"] = {}
            for target_len in ctx_lengths:
                doc, codes = make_needle_doc(target_len, seed=target_len, nonce=f"{engine}-ctx")
                out = sse_request(
                    args.port,
                    "/v1/chat/completions",
                    {
                        "model": model_name,
                        "messages": [{"role": "user", "content": doc}],
                        "max_tokens": 128,
                        "temperature": 0.0,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "cache_prompt": False,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    timeout_s=900.0,
                )
                found = sum(1 for code in codes if code in out["content"])
                key = str(target_len)
                res["long_ctx"][key] = {
                    "target_tokens": target_len,
                    "prompt_tokens": out["prompt_tokens"],
                    "completion_tokens": out["completion_tokens"],
                    "ttft_s": round(out["ttft_ms"] / 1000.0, 2),
                    "client_prefill_tps": out["client_prefill_tps"],
                    "srv_prefill_tps": out["srv_prefill_tps"],
                    "client_decode_tps": out["client_decode_tps"],
                    "srv_decode_tps": out["srv_decode_tps"],
                    "needles": f"{found}/3",
                    "needles_ok": found == 3,
                }
                print(
                    f"  [{engine}] ctx_{target_len} ({out['prompt_tokens']} tok): "
                    f"ttft={out['ttft_ms']/1000.0:.2f}s ({out['client_prefill_tps']:.1f} tok/s) | "
                    f"decode={out['client_decode_tps']:.2f} tok/s | needles={found}/3",
                    flush=True,
                )
    finally:
        if vram_mon is not None:
            vstats = vram_mon.stop()
            res.setdefault("system", {}).update(vstats)
        srv.stop()

    return res


def fmt_delta(base: float, cand: float, higher_is_better: bool = True) -> str:
    if base <= 0:
        return "—"
    pct = (cand - base) / base * 100.0
    sign = "+" if pct >= 0 else ""
    good = (pct >= 0) if higher_is_better else (pct <= 0)
    tag = f"{sign}{pct:.1f}%"
    return f"**{tag}**" if good else tag


def fmt_ratio(base: float, cand: float, higher_is_better: bool = True) -> str:
    if base <= 0 or cand <= 0:
        return "—"
    if higher_is_better:
        pct = (cand - base) / base * 100.0
        sign = "+" if pct >= 0 else ""
        return f"**{sign}{pct:.1f}%**" if pct >= 0 else f"{sign}{pct:.1f}%"
    else:
        ratio = base / cand
        pct = (cand - base) / base * 100.0
        sign = "+" if pct >= 0 else ""
        txt = f"{sign}{pct:.1f}% ({ratio:.2f}×)"
        return f"**{txt}**" if pct <= 0 else txt


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def render_markdown(data: dict[str, Any]) -> str:
    gpu = data["gpu"]
    L = data.get("llama")
    Z = data.get("zig")
    lines: list[str] = []
    lines.append("# Head-to-Head Benchmark: `llama-server` (Tuned) vs `kintsugi-zig`")
    lines.append("")
    lines.append(
        f"- **GPU**: {gpu['name']} ({gpu['memory_total_mib']} MiB VRAM, driver {gpu['driver_version']})"
    )
    lines.append(f"- **Model**: `Kintsugi-27B-12G.gguf` + `dflash-Qwen3.8-27B-Q4_0.gguf`")
    lines.append(
        "- **`llama-server` Config**: Tuned `best-v2` (`LLAMA_HEAD2=64`, `--kv-vram-cells 69632`, "
        "`--cache-ram 6144`, `-fa on`, `-ctk/-ctv q4_0`, `--spec-type draft-dflash --spec-draft-n-max 7`)"
    )
    lines.append(
        f"- **`kintsugi-zig` Config**: Commit `{data.get('zig_commit', 'unknown')}` "
        "(native `sm_89` CUBIN, INT8 Tensor Core Escha + MMA6x8 FA, `PREFILL_M=240`, 5-bucket Split-K, Parallel CUDA Graph, Adaptive PLD + Top-2 Tree Spec)"
    )
    lines.append(
        f"- **Methodology**: Sequential runs on idle GPU (0 contention), SSE client-measured TTFT & "
        f"`(completion_tokens - 1) / (t_end - t_first)` decode tok/s, median of {data['runs']} cold runs."
    )
    lines.append("")

    if not (L and Z):
        lines.append("_Only one engine was run; pass `--only both` for side-by-side deltas._")
        return "\n".join(lines)

    # 1. System & Capacity
    lines.append("## 1. Startup, VRAM & Capacity")
    lines.append("")
    lines.append("| Metric | `llama-server` (Tuned) | `kintsugi-zig` | Delta (`zig` vs `llama`) |")
    lines.append("| :--- | ---: | ---: | ---: |")
    ls, zs = L["system"], Z["system"]
    lines.append(
        f"| Cold Startup Time (spawn → `/health` 200) | {ls['startup_s']:.2f} s | **{zs['startup_s']:.2f} s** | "
        f"{fmt_ratio(ls['startup_s'], zs['startup_s'], higher_is_better=False)} |"
    )
    lines.append(
        f"| Idle Process VRAM | {ls['idle_vram_mib']} MiB | {zs['idle_vram_mib']} MiB | "
        f"{fmt_delta(ls['idle_vram_mib'], zs['idle_vram_mib'], higher_is_better=False)} |"
    )
    lines.append(
        f"| Peak Process VRAM (full suite) | {ls['peak_proc_mib']} MiB | {zs['peak_proc_mib']} MiB | "
        f"{fmt_delta(ls['peak_proc_mib'], zs['peak_proc_mib'], higher_is_better=False)} |"
    )
    lines.append(
        f"| Min Free GPU VRAM Headroom | {ls['min_free_mib']} MiB | {zs['min_free_mib']} MiB | "
        f"{zs['min_free_mib'] - ls['min_free_mib']:+d} MiB |"
    )
    lines.append(
        f"| In-VRAM Q4_0 KV Cache Pool | {ls['kv_vram_cells']:,} cells | **{zs['kv_vram_cells']:,} cells** | "
        f"{fmt_delta(ls['kv_vram_cells'], zs['kv_vram_cells'], higher_is_better=True)} |"
    )
    if "m1_baseline" in L and "m1_baseline" in Z:
        lm1, zm1 = L["m1_baseline"], Z["m1_baseline"]
        lines.append(
            f"| Pure `M=1` Decode (No-Spec Baseline) | {lm1['client_decode_tps']:.2f} tok/s | "
            f"**{zm1['client_decode_tps']:.2f} tok/s** | "
            f"{fmt_delta(lm1['client_decode_tps'], zm1['client_decode_tps'], higher_is_better=True)} |"
        )
    lines.append("")

    # 2. Short-Context Speculative Decode
    lines.append("## 2. Short-Context Speculative Decode (`/v1/completions`, `max_tokens=256`)")
    lines.append("")
    lines.append(
        "| Workload | `llama` Decode | `zig` Decode | Decode Delta | `llama` TTFT | `zig` TTFT | TTFT Delta |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for c in SHORT_CASES:
        cid = c["id"]
        lc, zc = L["short"][cid], Z["short"][cid]
        lines.append(
            f"| {c['label']} | {lc['client_decode_tps']:.2f} tok/s | **{zc['client_decode_tps']:.2f} tok/s** | "
            f"{fmt_delta(lc['client_decode_tps'], zc['client_decode_tps'], True)} | "
            f"{lc['ttft_ms']:.1f} ms | {zc['ttft_ms']:.1f} ms | "
            f"{fmt_ratio(lc['ttft_ms'], zc['ttft_ms'], False)} |"
        )
    lines.append("")

    # 3. Chat API
    lines.append("## 3. Chat Completions API (`/v1/chat/completions`, Streaming SSE)")
    lines.append("")
    lines.append("| Mode | `llama` Decode | `zig` Decode | Decode Delta | `llama` TTFT | `zig` TTFT | Correctness |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | :---: |")
    lcd, zcd = L["chat"]["chat_direct"], Z["chat"]["chat_direct"]
    lines.append(
        f"| Thinking OFF (ZH Direct, 256 tok) | {lcd['client_decode_tps']:.2f} tok/s | "
        f"**{zcd['client_decode_tps']:.2f} tok/s** | "
        f"{fmt_delta(lcd['client_decode_tps'], zcd['client_decode_tps'], True)} | "
        f"{lcd['ttft_ms']:.1f} ms | {zcd['ttft_ms']:.1f} ms | Both `stop`/`length` |"
    )
    lct, zct = L["chat"]["chat_think"], Z["chat"]["chat_think"]
    lines.append(
        f"| Thinking ON (`17*29+143=636`) | {lct['client_decode_tps']:.2f} tok/s | "
        f"**{zct['client_decode_tps']:.2f} tok/s** | "
        f"{fmt_delta(lct['client_decode_tps'], zct['client_decode_tps'], True)} | "
        f"{lct['ttft_ms']:.1f} ms | {zct['ttft_ms']:.1f} ms | "
        f"llama={lct['correct_636']}, zig={zct['correct_636']} |"
    )
    lines.append("")

    # 4. Long Context & Prefix Cache
    if "long_ctx" in L and "long_ctx" in Z:
        lines.append("## 4. Long-Context Prefill, Decode & 3-Needle Recall (`/v1/chat/completions`, Cold)")
        lines.append("")
        lines.append(
            "| Context | Prompt Tok | `llama` Prefill | `zig` Prefill | Prefill Delta | "
            "`llama` Decode | `zig` Decode | Decode Delta | Needles (L / Z) |"
        )
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |")
        for k in L["long_ctx"]:
            if k not in Z["long_ctx"]:
                continue
            lx, zx = L["long_ctx"][k], Z["long_ctx"][k]
            label = f"{int(k)//1024}K" if int(k) % 1024 == 0 else f"{int(k)/1000:.0f}K"
            if int(k) > 69632:
                label += " *(llama Host-RAM tier)*"
            lines.append(
                f"| {label} | {zx['prompt_tokens']:,} | "
                f"{lx['client_prefill_tps']:.1f} tok/s ({lx['ttft_s']:.2f}s) | "
                f"**{zx['client_prefill_tps']:.1f} tok/s** ({zx['ttft_s']:.2f}s) | "
                f"{fmt_delta(lx['client_prefill_tps'], zx['client_prefill_tps'], True)} | "
                f"{lx['client_decode_tps']:.2f} tok/s | **{zx['client_decode_tps']:.2f} tok/s** | "
                f"{fmt_delta(lx['client_decode_tps'], zx['client_decode_tps'], True)} | "
                f"{lx['needles']} / {zx['needles']} |"
            )
        lines.append("")

    if "prefix_cache_10k" in L and "prefix_cache_10k" in Z:
        lp, zp = L["prefix_cache_10k"], Z["prefix_cache_10k"]
        lines.append("## 5. Multi-Turn Prefix Cache Reuse (~10K Shared System Prompt)")
        lines.append("")
        lines.append("| Metric | `llama-server` (`--cache-ram 6144`) | `kintsugi-zig` (8 Host Checkpoints) | Delta |")
        lines.append("| :--- | ---: | ---: | ---: |")
        lines.append(
            f"| Turn 1 Cold TTFT ({zp['prompt_tokens']:,} tok) | {lp['turn1_ttft_ms']:.1f} ms | "
            f"**{zp['turn1_ttft_ms']:.1f} ms** | {fmt_ratio(lp['turn1_ttft_ms'], zp['turn1_ttft_ms'], False)} |"
        )
        lines.append(
            f"| Turn 2 Cached TTFT | {lp['turn2_ttft_ms']:.1f} ms | "
            f"**{zp['turn2_ttft_ms']:.1f} ms** | {fmt_ratio(lp['turn2_ttft_ms'], zp['turn2_ttft_ms'], False)} |"
        )
        lines.append(
            f"| Cache Speedup (T1 / T2) | {lp['speedup_x']:.1f}× | **{zp['speedup_x']:.1f}×** | — |"
        )
        lines.append(
            f"| Needle Recall (T1 / T2) | {lp['turn1_hit']} / {lp['turn2_hit']} | "
            f"{zp['turn1_hit']} / {zp['turn2_hit']} | — |"
        )
        lines.append("")

    # 6. Greedy Output Parity
    lines.append("## 6. Greedy Output Parity Check (`T=0.0`, Raw `/v1/completions`)")
    lines.append("")
    lines.append("| Case | Exact Match | Common Prefix / Total Chars |")
    lines.append("| :--- | :---: | ---: |")
    for c in SHORT_CASES:
        if not c.get("parity"):
            continue
        cid = c["id"]
        a = L["short"][cid]["sample_text"]
        b = Z["short"][cid]["sample_text"]
        cpl = common_prefix_len(a, b)
        exact = a == b
        lines.append(f"| `{cid}` | {'YES' if exact else 'DIFF'} | {cpl} / {min(len(a), len(b))} chars |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark llama-server vs kintsugi-zig head-to-head")
    ap.add_argument("--only", choices=("both", "llama", "zig"), default="both")
    ap.add_argument("--runs", type=int, default=2, help="Measured cold runs per short workload (default: 2)")
    ap.add_argument("--skip-long", action="store_true", help="Skip 10K prefix cache and long-context benchmarks")
    ap.add_argument("--skip-m1", action="store_true", help="Skip pure M=1 (no-spec) baseline")
    ap.add_argument(
        "--ctx-lengths",
        default="2048,10240,20480,76000",
        help="Comma-separated long-context token targets (default: 2048,10240,20480,76000)",
    )
    ap.add_argument("--port", type=int, default=18081)
    ap.add_argument("--zig-bin", type=Path, default=DEFAULT_ZIG_BIN)
    ap.add_argument("--llama-root", type=Path, default=DEFAULT_LLAMA_ROOT)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--drafter", type=Path, default=DEFAULT_DRAFTER)
    ap.add_argument("--out-json", type=Path, default=REPO_ROOT / "bench_vs_llama_results.json")
    ap.add_argument("--out-md", type=Path, default=REPO_ROOT / "bench_vs_llama_report.md")
    args = ap.parse_args()

    ctx_lengths = [int(x.strip()) for x in args.ctx_lengths.split(",") if x.strip()]
    gpu = query_gpu_info()
    print(f"GPU: {gpu['name']} | Free VRAM: {gpu['memory_free_mib']} / {gpu['memory_total_mib']} MiB", flush=True)

    try:
        zig_commit = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        zig_commit = "unknown"

    log_dir = Path("/tmp/kintsugi-bench-logs")
    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": gpu,
        "runs": args.runs,
        "ctx_lengths": ctx_lengths,
        "zig_commit": zig_commit,
    }

    engines = ("llama", "zig") if args.only == "both" else (args.only,)
    for eng in engines:
        results[eng] = run_engine_suite(eng, args, ctx_lengths, log_dir)

    args.out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    md = render_markdown(results)
    args.out_md.write_text(md)
    print("\n" + "=" * 72)
    print(md)
    print("=" * 72)
    print(f"Saved JSON: {args.out_json}")
    print(f"Saved MD:   {args.out_md}")


if __name__ == "__main__":
    main()
