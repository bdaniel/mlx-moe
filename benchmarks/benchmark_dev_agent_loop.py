#!/usr/bin/env python3
"""Multi-turn, execution-graded benchmark for local developer agents."""

from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx_lm
import httpx

from mlx_moe.lazy_experts.core import get_faithful_stats, get_fallback_stats
from mlx_moe.lazy_experts.generate import (
    _load_model_and_tokenizer,
    _startup,
    _stream_with_dynamic_updates,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "logs" / "dev-agent-loop"


@dataclass(frozen=True)
class Scenario:
    name: str
    category: str
    issue: str
    files: dict[str, str]
    hidden_test: str


SCENARIOS = (
    Scenario(
        "timeout_zero",
        "multi_file_repair",
        "Users report timeout=0 is ignored. Diagnose and fix with minimal scope. "
        "Run tests, inspect callers as needed, and finish only when confident.",
        {
            "app/__init__.py": "",
            "app/config.py": "DEFAULT_TIMEOUT = 5\n",
            "app/client.py": (
                "from .config import DEFAULT_TIMEOUT\n\n"
                "def effective_timeout(timeout):\n"
                "    return timeout or DEFAULT_TIMEOUT\n\n"
                "def request_options(timeout=None):\n"
                "    return {\"timeout\": effective_timeout(timeout)}\n"
            ),
            "app/retry.py": (
                "def attempts(retries):\n"
                "    return list(range(retries + 1))\n"
            ),
            "tests/test_client.py": (
                "import unittest\n"
                "from app.client import effective_timeout\n\n"
                "class ClientTests(unittest.TestCase):\n"
                "    def test_default(self):\n"
                "        self.assertEqual(effective_timeout(None), 5)\n"
                "    def test_explicit(self):\n"
                "        self.assertEqual(effective_timeout(12), 12)\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        """import unittest
from app.client import effective_timeout, request_options
from app.retry import attempts

class HiddenTests(unittest.TestCase):
    def test_zero_disables_timeout(self):
        self.assertEqual(effective_timeout(0), 0)
        self.assertEqual(request_options(0), {"timeout": 0})
    def test_defaults_and_unrelated_behavior(self):
        self.assertEqual(effective_timeout(None), 5)
        self.assertEqual(attempts(2), [0, 1, 2])

if __name__ == "__main__":
    unittest.main()
""",
    ),
    Scenario(
        "protocol_decoder",
        "reverse_engineering",
        "Captured frames no longer decode after a firmware update. Infer format from "
        "captures.txt, repair decoder, and run tests. Preserve strict malformed-frame rejection.",
        {
            "protocol/__init__.py": "",
            "protocol/decoder.py": (
                "MAGIC = bytes.fromhex('aa55')\n\n"
                "def decode_frame(data: bytes) -> bytes:\n"
                "    if len(data) < 5 or data[:2] != MAGIC:\n"
                "        raise ValueError('bad frame')\n"
                "    size = int.from_bytes(data[2:4], 'big')\n"
                "    if len(data) != 5 + size:\n"
                "        raise ValueError('bad length')\n"
                "    payload = data[4:-1]\n"
                "    if sum(payload) & 0xff != data[-1]:\n"
                "        raise ValueError('bad checksum')\n"
                "    return payload\n"
            ),
            "captures.txt": (
                "aa55030041424340 -> payload ASCII ABC\n"
                "aa550200000000 -> payload 00 00\n"
                "aa5504000102030404 -> payload 01 02 03 04\n"
                "aa550100ff00 -> rejected: checksum mismatch\n"
            ),
            "tests/test_decoder.py": (
                "import unittest\n"
                "from protocol.decoder import decode_frame\n\n"
                "class DecoderTests(unittest.TestCase):\n"
                "    def test_empty(self):\n"
                "        self.assertEqual(decode_frame(bytes.fromhex('aa55000000')), b'')\n"
                "    def test_magic(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            decode_frame(bytes.fromhex('0055000000'))\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        """import unittest
from protocol.decoder import decode_frame

class HiddenTests(unittest.TestCase):
    def test_captures(self):
        self.assertEqual(decode_frame(bytes.fromhex("aa55030041424340")), b"ABC")
        self.assertEqual(decode_frame(bytes.fromhex("aa5504000102030404")), b"\x01\x02\x03\x04")
    def test_rejects_bad_frames(self):
        bad = ("aa550100ff00", "aa55030041424341", "aa55040041424340", "aa5503004142434000")
        for value in bad:
            with self.assertRaises(ValueError):
                decode_frame(bytes.fromhex(value))

if __name__ == "__main__":
    unittest.main()
""",
    ),
    Scenario(
        "async_singleflight",
        "concurrency_debugging",
        "Cache sometimes deadlocks under load. Fix deadlock without allowing duplicate "
        "fetches for the same key. Failed fetches must remain retryable. Run tests.",
        {
            "service/__init__.py": "",
            "service/cache.py": (
                "import asyncio\n\n"
                "class AsyncCache:\n"
                "    def __init__(self):\n"
                "        self._lock = asyncio.Lock()\n"
                "        self._values = {}\n\n"
                "    async def size(self):\n"
                "        async with self._lock:\n"
                "            return len(self._values)\n\n"
                "    async def get_or_fetch(self, key, fetch):\n"
                "        async with self._lock:\n"
                "            if key not in self._values:\n"
                "                self._values[key] = await fetch()\n"
                "            return self._values[key]\n"
            ),
            "tests/test_cache.py": (
                "import asyncio\n"
                "import unittest\n"
                "from service.cache import AsyncCache\n\n"
                "class CacheTests(unittest.IsolatedAsyncioTestCase):\n"
                "    async def test_fetch_can_inspect_cache(self):\n"
                "        cache = AsyncCache()\n"
                "        async def fetch():\n"
                "            await cache.size()\n"
                "            return 7\n"
                "        value = await asyncio.wait_for(cache.get_or_fetch('x', fetch), .2)\n"
                "        self.assertEqual(value, 7)\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        """import asyncio
import unittest
from service.cache import AsyncCache

class HiddenTests(unittest.IsolatedAsyncioTestCase):
    async def test_singleflight(self):
        cache = AsyncCache()
        calls = 0
        async def fetch():
            nonlocal calls
            calls += 1
            await asyncio.sleep(.02)
            return 9
        values = await asyncio.wait_for(
            asyncio.gather(*(cache.get_or_fetch("x", fetch) for _ in range(8))), 1)
        self.assertEqual(values, [9] * 8)
        self.assertEqual(calls, 1)

    async def test_failure_is_retryable(self):
        cache = AsyncCache()
        calls = 0
        async def fetch():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary")
            return 11
        with self.assertRaises(RuntimeError):
            await cache.get_or_fetch("x", fetch)
        self.assertEqual(await cache.get_or_fetch("x", fetch), 11)

if __name__ == "__main__":
    unittest.main()
""",
    ),
)


SYSTEM = """You are an autonomous coding agent in a disposable Python repository.
Solve issue using tools. Inspect before editing. Run tests after edits. Add regression tests for explicit requirements not covered by existing tests. Do not repeat a successful read/search unless files changed. When evidence is sufficient, edit instead of re-inspecting. Call exactly one tool per turn, with no prose.
Never access paths outside repository. Do not finish before tests pass.
"""


def _function(name: str, description: str, properties: dict | None = None, required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": list(required),
            },
        },
    }


TOOLS = [
    _function("list_files", "List repository files."),
    _function(
        "read_file", "Read numbered lines from a repository file.",
        {"path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}},
        ("path",),
    ),
    _function("search_code", "Search repository text.", {"query": {"type": "string"}}, ("query",)),
    _function("symbols", "List Python symbols in a file.", {"path": {"type": "string"}}, ("path",)),
    _function("references", "Find references to a Python identifier.", {"name": {"type": "string"}}, ("name",)),
    _function(
        "write_file", "Replace a Python file with complete content.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ("path", "content"),
    ),
    _function(
        "replace_text", "Replace one exact text occurrence in a Python file.",
        {
            "path": {"type": "string"},
            "old": {"type": "string"},
            "new": {"type": "string"},
        },
        ("path", "old", "new"),
    ),
    _function(
        "run_tests", "Run all repository unit tests, or one Python test file.",
        {"path": {"type": "string"}},
    ),
    _function("finish", "Finish after tests pass.", {"summary": {"type": "string"}}, ("summary",)),
]


def _json_value(text: str) -> dict:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object found")


def _normalize_action(value: dict) -> dict:
    if "action" in value:
        return value
    if isinstance(value.get("function"), dict):
        value = value["function"]
    name = value.get("name")
    if not name:
        raise ValueError("tool response has no action or name")
    arguments = value.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")
    return {"action": name, **arguments}


def _gpt_oss_action(text: str) -> dict | None:
    patterns = (
        r"assistant to=functions\.([A-Za-z_][A-Za-z0-9_]*)"
        r"<\|channel\|>commentary(?: [^<]*)?<\|message\|>(.*?)(?:<\|call\|>|$)",
        r"<\|channel\|>(?:commentary|analysis) to=functions\.([A-Za-z_][A-Za-z0-9_]*)"
        r"(?:\s*<\|constrain\|>[^<]*)*<\|message\|>(.*?)(?:<\|call\|>|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match is not None:
            arguments = _json_value(match.group(2))
            return {"action": match.group(1), **arguments}
    return None


def _assistant_history(text: str) -> dict:
    thinking_matches = re.findall(
        r"<\|channel\|>analysis<\|message\|>(.*?)(?=<\|end\|>|<\|return\|>|$)",
        text,
        re.DOTALL,
    )
    final_matches = re.findall(
        r"<\|channel\|>final<\|message\|>(.*?)(?=<\|end\|>|<\|return\|>|$)",
        text,
        re.DOTALL,
    )
    message = {"role": "assistant", "content": final_matches[-1] if final_matches else ""}
    if thinking_matches:
        message["thinking"] = thinking_matches[-1]
    elif not final_matches:
        message["content"] = text
    return message


def _action_value(tokenizer, text: str) -> dict:
    if action := _gpt_oss_action(text):
        return action
    final = re.search(
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
        text,
        re.DOTALL,
    )
    if final is not None:
        return {"action": "finish", "summary": final.group(1).strip()}
    start = tokenizer.tool_call_start
    end = tokenizer.tool_call_end
    if start and end and start in text:
        body = text.split(start, 1)[1].split(end, 1)[0]
        parsed = tokenizer.tool_parser(body, TOOLS)
        if isinstance(parsed, list):
            parsed = parsed[0]
        return _normalize_action(parsed)
    return _normalize_action(_json_value(text))


def _chat_prompt(
    tokenizer,
    messages: list[dict],
    enable_thinking: bool = False,
    tools: list | None = TOOLS,
) -> str:
    if not tokenizer.has_chat_template:
        content = "\n\n".join(str(message.get("content", "")) for message in messages)
        return f"{SYSTEM}\n\n{content}"
    kwargs = {"add_generation_prompt": True, "tokenize": False}
    if tokenizer.has_thinking:
        kwargs["enable_thinking"] = enable_thinking
    if tools is not None:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)


def _local_generate(
    model,
    tokenizer,
    messages: list[dict],
    *,
    tools: list | None,
    enable_thinking: bool,
    max_tokens: int,
    backend: str,
    args,
) -> tuple[str, float]:
    prompt = _chat_prompt(tokenizer, messages, enable_thinking, tools)
    prefill_step_size = (
        16
        if backend == "mlx-moe" and args.expert_policy == "faithful"
        else args.prefill_step_size
    )
    kwargs = {"prefill_step_size": prefill_step_size}
    model_type = getattr(model, "model_type", None)
    if args.kv_bits is not None and model_type != "gpt_oss":
        kwargs.update(kv_bits=args.kv_bits, kv_group_size=64, quantized_kv_start=0)
    responses = mlx_lm.stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, **kwargs
    )
    if backend == "mlx-moe":
        responses = _stream_with_dynamic_updates(
            model,
            responses,
            adaptive_growth=args.adaptive_growth,
        )
    text = ""
    last = None
    for response in responses:
        text += response.text
        last = response
    return text, last.generation_tps if last is not None else 0.0


def _api_generate(
    url: str,
    model: str,
    messages: list[dict],
    *,
    tools: list | None,
    enable_thinking: bool,
    max_tokens: int,
    cache_key: str,
    timeout: float,
) -> tuple[str, dict | None]:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "cache_key": cache_key,
        "enable_thinking": enable_thinking,
    }
    if tools is not None:
        body["tools"] = tools
    response = httpx.post(url, json=body, timeout=timeout)
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    return message.get("content") or "", message


def _inside(root: Path, raw: str) -> Path:
    root = root.resolve()
    path = (root / raw).resolve()
    if path != root and root not in path.parents:
        raise ValueError("path outside repository")
    return path


def _run(command: list[str], cwd: Path, timeout: float = 8) -> dict:
    try:
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"exit_code": 124, "output": "TIMEOUT"}
    output = (result.stdout + result.stderr)[-6000:]
    return {"exit_code": result.returncode, "output": output}


def _tool(action: dict, root: Path) -> dict:
    root = root.resolve()
    name = action.get("action")
    if name == "list_files":
        files = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())
        return {"files": files}
    if name == "read_file":
        path = _inside(root, str(action.get("path", "")))
        lines = path.read_text().splitlines()
        start = max(1, int(action.get("start", 1)))
        end = min(len(lines), int(action.get("end", start + 239)))
        return {"content": "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))}
    if name == "search_code":
        query = str(action.get("query", ""))
        if not query:
            raise ValueError("empty query")
        matches = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".txt", ".md"}:
                continue
            for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if query.lower() in line.lower():
                    matches.append(f"{path.relative_to(root)}:{line_no}:{line}")
        return {"matches": matches[:100]}
    if name == "symbols":
        path = _inside(root, str(action.get("path", "")))
        tree = ast.parse(path.read_text())
        symbols = [
            {"name": node.name, "kind": type(node).__name__, "line": node.lineno}
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        return {"symbols": symbols}
    if name == "references":
        identifier = str(action.get("name", ""))
        if not identifier.isidentifier():
            raise ValueError("name must be identifier")
        pattern = re.compile(rf"\b{re.escape(identifier)}\b")
        matches = []
        for path in sorted(root.rglob("*.py")):
            for line_no, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line):
                    matches.append(f"{path.relative_to(root)}:{line_no}:{line}")
        return {"references": matches[:100]}
    if name == "write_file":
        path = _inside(root, str(action.get("path", "")))
        if path.suffix != ".py":
            raise ValueError("only Python files may be written")
        content = action.get("content")
        if not isinstance(content, str):
            raise ValueError("content must be string")
        ast.parse(content)
        if path.exists() and path.read_text() == content:
            return {
                "error": "file content unchanged; inspect evidence or make a real edit",
                "written": str(path.relative_to(root)),
                "bytes": len(content.encode()),
                "changed": False,
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return {
            "written": str(path.relative_to(root)),
            "bytes": len(content.encode()),
            "changed": True,
        }
    if name == "replace_text":
        path = _inside(root, str(action.get("path", "")))
        if path.suffix != ".py":
            raise ValueError("only Python files may be written")
        old = action.get("old", action.get("content"))
        new = action.get("new")
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            raise ValueError("old and new must be strings; old must not be empty")
        content = path.read_text()
        count = content.count(old)
        if count != 1:
            raise ValueError(f"old text must occur exactly once; found {count}")
        if old == new:
            return {
                "error": "replacement leaves file unchanged",
                "written": str(path.relative_to(root)),
                "changed": False,
            }
        updated = content.replace(old, new, 1)
        ast.parse(updated)
        path.write_text(updated)
        return {
            "written": str(path.relative_to(root)),
            "bytes": len(updated.encode()),
            "changed": True,
        }
    if name == "run_tests":
        raw_path = action.get("path")
        if raw_path:
            path = _inside(root, str(raw_path))
            if path.suffix != ".py":
                raise ValueError("test path must be a Python file")
            return _run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    str(path.parent.relative_to(root)),
                    "-p",
                    path.name,
                    "-v",
                ],
                root,
            )
        return _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], root)
    if name == "finish":
        return {"finished": True}
    raise ValueError(f"unknown action: {name}")


def _hidden_grade(root: Path, hidden_test: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as file:
        file.write(hidden_test)
        hidden_path = Path(file.name)
    try:
        env = dict(os.environ, PYTHONPATH=str(root))
        result = subprocess.run(
            [sys.executable, str(hidden_path)], cwd=root, env=env,
            text=True, capture_output=True, timeout=12,
        )
        output = (result.stdout + result.stderr)[-6000:]
        return result.returncode == 0, output
    finally:
        hidden_path.unlink(missing_ok=True)


def _free_percent() -> int | None:
    try:
        output = subprocess.check_output(["memory_pressure"], text=True, timeout=3)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    match = re.search(r"free percentage:\s*(\d+)%", output)
    return int(match.group(1)) if match else None


def _memory_guard(
    min_free_percent: int,
    *,
    attempts: int,
    interval: float,
    reclaim: bool,
) -> dict:
    readings = [_free_percent()]
    first = readings[0]
    if first is None or first >= min_free_percent:
        return {"safe": True, "readings": readings, "reclaimed": False}

    if reclaim:
        gc.collect()
        mx.clear_cache()

    for _ in range(1, attempts):
        if interval:
            time.sleep(interval)
        readings.append(_free_percent())
        if readings[-1] is None or readings[-1] >= min_free_percent:
            return {"safe": True, "readings": readings, "reclaimed": reclaim}

    return {"safe": False, "readings": readings, "reclaimed": reclaim}


def _retrieved_context(root: Path, query: str, max_chars: int) -> str:
    terms = set(re.findall(r"[a-z_][a-z0-9_]{3,}", query.lower()))
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix not in {".py", ".txt", ".md"}:
            continue
        content = path.read_text(errors="replace")
        relative = str(path.relative_to(root))
        haystack = f"{relative}\n{content}".lower()
        score = sum(haystack.count(term) for term in terms)
        if path.name.startswith("test_"):
            score += 1
        candidates.append((score, relative, content))
    candidates.sort(key=lambda item: (-item[0], item[1]))

    sections = ["FILES: " + ", ".join(item[1] for item in candidates)]
    remaining = max_chars - len(sections[0])
    for _, relative, content in candidates:
        section = f"\n\nFILE {relative}\n{content}"
        if remaining <= 0:
            break
        sections.append(section[:remaining])
        remaining -= len(section)
    return "".join(sections)


PLANNER_SYSTEM = """You are a software-engineering planner. Analyze supplied issue and repository evidence deeply. Verify arithmetic and byte-level claims explicitly. Do not call tools and do not write code as a complete file. Return a concise implementation plan containing likely root cause, exact edits, edge cases, and tests. Your plan is advisory; a separate execution agent will inspect and apply it."""


def _planner_messages(scenario: Scenario, max_chars: int) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix=f"planner-{scenario.name}-") as temp:
        root = Path(temp)
        for relative, content in scenario.files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        context = _retrieved_context(root, scenario.issue, max_chars)
    return [
        {"role": "system", "content": PLANNER_SYSTEM},
        {
            "role": "user",
            "content": f"ISSUE:\n{scenario.issue}\n\nREPOSITORY CONTEXT:\n{context}",
        },
    ]


def _trim_planner_repetition(text: str) -> tuple[str, bool]:
    repeated = re.search(r"\b(\w+)(?:[\s,]+\1\b){7,}", text, re.IGNORECASE)
    if repeated is None:
        return text, False
    return text[:repeated.start()].rstrip(), True


def _generate_plans(model, tokenizer, selected: list[Scenario], backend: str, args):
    plans = {}
    for scenario in selected:
        messages = _planner_messages(scenario, args.retrieval_chars)
        started = time.perf_counter()
        if backend == "openai":
            text, _ = _api_generate(
                args.planner_url,
                args.planner_model or args.model,
                messages,
                tools=None,
                enable_thinking=True,
                max_tokens=args.planner_max_tokens,
                cache_key=f"dev-planner-{scenario.name}",
                timeout=args.api_timeout,
            )
            tps = 0.0
        else:
            text, tps = _local_generate(
                model,
                tokenizer,
                messages,
                tools=None,
                enable_thinking=True,
                max_tokens=args.planner_max_tokens,
                backend=backend,
                args=args,
            )
        text, repetition_trimmed = _trim_planner_repetition(text)
        plans[scenario.name] = {
            "text": text[-args.planner_context_chars:],
            "seconds": time.perf_counter() - started,
            "generation_tps": tps,
            "repetition_trimmed": repetition_trimmed,
        }
        print(
            f"  planner {scenario.name}: {len(text)} chars "
            f"{plans[scenario.name]['seconds']:.1f}s {tps:.2f} tok/s"
            f"{' repetition-trimmed' if repetition_trimmed else ''}",
            flush=True,
        )
        if model is not None:
            gc.collect()
            mx.clear_cache()
    return plans


def _write_results(run_dir: Path, result: dict) -> None:
    (run_dir / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Developer Agent Loop Benchmark", "",
        f"- Model: `{result['model']}`",
        f"- Backend: `{result['backend']}`",
        f"- Capacity: `{result['capacity'] or 'n/a'}`",
        f"- Expert policy: `{result['expert_policy'] or 'n/a'}`",
        f"- Context mode: `{result['context_mode']}`",
        f"- Thinking strategy: `{result['thinking_strategy']}`",
        f"- Planner: `{result['planner_model'] or 'n/a'}`",
        f"- Status: `{result['status']}`",
        f"- Score: {result['score']:.1%}",
        f"- Mean turns: {result['mean_turns']:.1f}",
        f"- Mean decode: {result['mean_generation_tps']:.2f} tok/s",
        f"- Peak memory: {result['peak_memory_gb']:.2f} GB", "",
        "| Scenario | Category | Pass | Turns | Tests | Decode tok/s | Fallback |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for row in result["runs"]:
        lines.append(
            f"| {row['scenario']} | {row['category']} | {'yes' if row['passed'] else 'no'} | "
            f"{row['turns']} | {'pass' if row['hidden_tests_passed'] else 'fail'} | "
            f"{row['generation_tps']:.2f} | {row['fallback_rate']:.1%} |"
        )
    (run_dir / "benchmark.md").write_text("\n".join(lines) + "\n")


def _summary(model: str, rows: list[dict], status: str, args) -> dict:
    return {
        "model": model,
        "backend": args.backend,
        "capacity": args.capacity,
        "expert_policy": args.expert_policy if args.backend == "mlx-moe" else None,
        "adaptive_growth": args.adaptive_growth,
        "context_mode": args.context_mode,
        "thinking_strategy": args.thinking_strategy,
        "planner_model": (
            args.planner_model or model
            if args.thinking_strategy == "planner"
            else None
        ),
        "status": status,
        "score": sum(row["passed"] for row in rows) / len(rows) if rows else 0.0,
        "mean_turns": sum(row["turns"] for row in rows) / len(rows) if rows else 0.0,
        "mean_generation_tps": (
            sum(row["generation_tps"] for row in rows) / len(rows) if rows else 0.0
        ),
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "runs": rows,
    }


def _run_scenario(
    model,
    tokenizer,
    scenario: Scenario,
    root: Path,
    args,
    planner: dict | None = None,
) -> dict:
    root = root.resolve()
    for relative, content in scenario.files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    transcript = []
    token_rates = []
    finished = False
    invalid_streak = 0
    duplicate_streak = 0
    file_version = 0
    seen_observations = set()
    fallback_before = get_fallback_stats(model) if model is not None else {
        "total_requests": 0, "total_fallbacks": 0
    }
    faithful_before = get_faithful_stats(model) if model is not None else {
        "requests": 0, "hits": 0
    }
    retrieved = (
        _retrieved_context(root, scenario.issue, args.retrieval_chars)
        if args.context_mode == "hybrid"
        else None
    )
    initial_content = f"ISSUE:\n{scenario.issue}\n"
    if retrieved:
        initial_content += f"\nRETRIEVED REPOSITORY CONTEXT:\n{retrieved}\n"
    if planner:
        initial_content += (
            "\nPLANNER ANALYSIS (advisory; verify against repository and tests):\n"
            f"{planner['text']}\n"
        )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": initial_content},
    ]
    memory_checks = []
    started = time.perf_counter()
    for turn in range(1, args.max_turns + 1):
        memory_check = _memory_guard(
            args.min_free_percent,
            attempts=args.memory_guard_checks,
            interval=args.memory_guard_interval,
            reclaim=model is not None,
        )
        memory_checks.append({"turn": turn, **memory_check})
        if not memory_check["safe"]:
            readings = ", ".join(f"{value}%" for value in memory_check["readings"])
            transcript.append({
                "turn": turn,
                "error": f"memory guard: sustained low memory ({readings})",
            })
            break
        history_size = args.history_turns * 2
        prompt_messages = (
            messages
            if len(messages) <= history_size + 2
            else messages[:2] + messages[-history_size:]
        )
        api_action = None
        if args.backend == "openai":
            text, message = _api_generate(
                args.api_url,
                args.model,
                prompt_messages,
                tools=TOOLS,
                enable_thinking=args.thinking_strategy == "always",
                max_tokens=args.max_tokens,
                cache_key=f"dev-agent-{scenario.name}",
                timeout=args.api_timeout,
            )
            if tool_calls := message.get("tool_calls"):
                function = tool_calls[0]["function"]
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                api_action = {"action": function["name"], **arguments}
        else:
            text, tps = _local_generate(
                model,
                tokenizer,
                prompt_messages,
                tools=TOOLS,
                enable_thinking=args.thinking_strategy == "always",
                max_tokens=args.max_tokens,
                backend=args.backend,
                args=args,
            )
            token_rates.append(tps)
        action = None
        try:
            action = api_action or (
                _normalize_action(_json_value(text))
                if tokenizer is None
                else _action_value(tokenizer, text)
            )
            signature = (file_version, json.dumps(action, sort_keys=True))
            observation = action.get("action") in {
                "list_files", "read_file", "search_code", "symbols", "references",
                "run_tests",
            }
            if observation and signature in seen_observations:
                error = "duplicate unchanged observation; choose a different action"
                if action.get("action") == "run_tests":
                    error = (
                        "tests already ran at this file version with the same result; "
                        "edit code or tests, inspect different evidence, or finish"
                    )
                result = {
                    "error": error
                }
                duplicate_streak += 1
            else:
                result = _tool(action, root)
                if observation:
                    seen_observations.add(signature)
                if action.get("action") in {"write_file", "replace_text"}:
                    if result["changed"]:
                        file_version += 1
                        duplicate_streak = 0
                    else:
                        duplicate_streak += 1
                else:
                    duplicate_streak = 0
            invalid_streak = 0
        except Exception as exc:
            action = action or {"raw": text}
            result = {"error": str(exc)}
            invalid_streak += 1
        transcript.append({"turn": turn, "action": action, "result": result})
        if "action" in action:
            call_id = f"call_{turn}"
            arguments = {key: value for key, value in action.items() if key != "action"}
            assistant_message = _assistant_history(text)
            assistant_message["tool_calls"] = [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": action["action"],
                    "arguments": arguments,
                },
            }]
            messages.extend([
                assistant_message,
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result),
                },
            ])
        else:
            messages.extend([
                _assistant_history(text),
                {
                    "role": "user",
                    "content": "Invalid tool response. Call exactly one available tool now.",
                },
            ])
        print(
            f"  {scenario.name} turn {turn}: {action.get('action', 'invalid')}"
            f"{' error=' + result['error'] if 'error' in result else ''}",
            flush=True,
        )
        if action.get("action") == "finish":
            finished = True
            break
        if invalid_streak >= args.max_invalid_actions:
            transcript.append({"turn": turn, "error": "invalid action stall"})
            break
        if duplicate_streak >= args.max_duplicate_actions:
            transcript.append({"turn": turn, "error": "duplicate action stall"})
            break
        gc.collect()
        if model is not None:
            mx.clear_cache()

    public = _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], root)
    hidden_ok, hidden_output = _hidden_grade(root, scenario.hidden_test)
    fallback_after = get_fallback_stats(model) if model is not None else fallback_before
    faithful_after = get_faithful_stats(model) if model is not None else faithful_before
    fallback_requests = (
        fallback_after["total_requests"] - fallback_before["total_requests"]
    )
    fallback_misses = (
        fallback_after["total_fallbacks"] - fallback_before["total_fallbacks"]
    )
    faithful_requests = faithful_after["requests"] - faithful_before["requests"]
    faithful_hits = faithful_after["hits"] - faithful_before["hits"]
    return {
        "scenario": scenario.name,
        "category": scenario.category,
        "passed": bool(finished and public["exit_code"] == 0 and hidden_ok),
        "finished": finished,
        "public_tests_passed": public["exit_code"] == 0,
        "hidden_tests_passed": hidden_ok,
        "turns": len(transcript),
        "seconds": time.perf_counter() - started,
        "generation_tps": sum(token_rates) / len(token_rates) if token_rates else 0.0,
        "fallback_rate": (
            fallback_misses / fallback_requests if fallback_requests else 0.0
        ),
        "expert_hit_rate": (
            faithful_hits / faithful_requests if faithful_requests else 0.0
        ),
        "active_memory_gb": mx.get_active_memory() / 1e9 if model is not None else 0.0,
        "peak_memory_gb": mx.get_peak_memory() / 1e9 if model is not None else 0.0,
        "free_percent_after": _free_percent(),
        "memory_checks": memory_checks,
        "public_test_output": public["output"],
        "hidden_test_output": hidden_output,
        "planner": planner,
        "transcript": transcript,
    }


def _load_runtime(model_name: str, backend: str, selected: list[Scenario], args, planner=False):
    if backend == "openai":
        return None, None
    if backend == "mlx-moe":
        startup_prompt = "\n".join(scenario.issue for scenario in selected)
        profile = args.profile if model_name == args.model else None
        return _startup(
            model_name,
            startup_prompt,
            profile_path=str(profile) if profile else None,
            capacity=(
                args.planner_capacity or args.capacity if planner else args.capacity
            ),
            warmup="none" if planner else args.warmup,
            pin_top_k=args.pin_top_k,
            expert_policy=args.expert_policy,
        )[:2]
    model, tokenizer, _ = _load_model_and_tokenizer(model_name, lazy=True)
    mx.eval(model.parameters())
    return model, tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--scenario", action="append", choices=tuple(s.name for s in SCENARIOS))
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--history-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument(
        "--thinking-strategy", choices=("off", "always", "planner"), default="off"
    )
    parser.add_argument("--planner-model")
    parser.add_argument(
        "--planner-backend",
        choices=("same", "mlx-lm", "mlx-moe", "openai"),
        default="same",
    )
    parser.add_argument("--planner-url")
    parser.add_argument("--planner-capacity", type=int)
    parser.add_argument("--planner-max-tokens", type=int, default=1024)
    parser.add_argument("--planner-context-chars", type=int, default=8000)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--soak-minutes", type=float)
    parser.add_argument("--kv-bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--min-free-percent", type=int, default=15)
    parser.add_argument("--memory-guard-checks", type=int, default=3)
    parser.add_argument("--memory-guard-interval", type=float, default=0.5)
    parser.add_argument("--max-invalid-actions", type=int, default=3)
    parser.add_argument("--max-duplicate-actions", type=int, default=4)
    parser.add_argument("--context-mode", choices=("tools", "hybrid"), default="tools")
    parser.add_argument("--retrieval-chars", type=int, default=6000)
    parser.add_argument(
        "--backend", choices=("mlx-lm", "mlx-moe", "openai"), default="mlx-lm"
    )
    parser.add_argument(
        "--api-url", default="http://127.0.0.1:8080/v1/chat/completions"
    )
    parser.add_argument("--api-timeout", type=float, default=600)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--pin-top-k", type=int)
    parser.add_argument("--warmup", choices=("hybrid", "full", "none"), default="none")
    parser.add_argument(
        "--expert-policy", choices=("approximate", "faithful"), default="approximate"
    )
    parser.add_argument("--adaptive-growth", action="store_true")
    args = parser.parse_args()

    selected = [s for s in SCENARIOS if not args.scenario or s.name in args.scenario]
    run_dir = args.results_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    mx.reset_peak_memory()
    planner_backend = (
        args.backend if args.planner_backend == "same" else args.planner_backend
    )
    planner_model_name = args.planner_model or args.model
    args.planner_url = args.planner_url or args.api_url
    plans = {}
    model = tokenizer = None
    if args.thinking_strategy == "planner":
        reuse_runtime = (
            planner_backend == args.backend and planner_model_name == args.model
        )
        if reuse_runtime:
            model, tokenizer = _load_runtime(
                args.model, args.backend, selected, args
            )
            plans = _generate_plans(
                model, tokenizer, selected, planner_backend, args
            )
        else:
            planner_model, planner_tokenizer = _load_runtime(
                planner_model_name, planner_backend, selected, args, planner=True
            )
            plans = _generate_plans(
                planner_model, planner_tokenizer, selected, planner_backend, args
            )
            planner_model = planner_tokenizer = None
            gc.collect()
            mx.clear_cache()
            print(f"  planner released; memory free={_free_percent()}%", flush=True)

    if model is None and args.backend != "openai":
        model, tokenizer = _load_runtime(args.model, args.backend, selected, args)

    rows = []
    deadline = time.monotonic() + args.soak_minutes * 60 if args.soak_minutes else None
    cycle = 0
    while cycle < args.repeat or (deadline is not None and time.monotonic() < deadline):
        cycle += 1
        for scenario in selected:
            if deadline is not None and rows and time.monotonic() >= deadline:
                break
            with tempfile.TemporaryDirectory(prefix=f"bonsai-{scenario.name}-") as temp:
                row = _run_scenario(
                    model,
                    tokenizer,
                    scenario,
                    Path(temp),
                    args,
                    plans.get(scenario.name),
                )
            row["cycle"] = cycle
            rows.append(row)
            result = _summary(args.model, rows, "running", args)
            _write_results(run_dir, result)
            print(
                f"[{cycle}:{scenario.name}] {'PASS' if row['passed'] else 'FAIL'} "
                f"turns={row['turns']} hidden={'pass' if row['hidden_tests_passed'] else 'fail'} "
                f"{row['generation_tps']:.2f} tok/s",
                flush=True,
            )
        if deadline is None and cycle >= args.repeat:
            break

    result = _summary(args.model, rows, "complete", args)
    _write_results(run_dir, result)
    print(f"Score: {result['score']:.1%}; results: {run_dir / 'benchmark.md'}")
    return 0 if result["score"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
