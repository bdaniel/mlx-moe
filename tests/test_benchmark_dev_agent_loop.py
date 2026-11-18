from benchmarks import benchmark_dev_agent_loop as benchmark


class _Tokenizer:
    tool_call_start = None
    tool_call_end = None


def test_gpt_oss_action_accepts_native_code_header():
    text = (
        "<|channel|>analysis<|message|>Run the focused test.<|end|>"
        "<|start|>assistant<|channel|>analysis to=functions.run_tests code"
        '<|message|>{"path":"tests/test_client.py"}'
    )

    assert benchmark._action_value(_Tokenizer(), text) == {
        "action": "run_tests",
        "path": "tests/test_client.py",
    }


def test_gpt_oss_action_accepts_native_json_header():
    text = (
        "<|channel|>analysis<|message|>Run tests.<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.run_tests"
        '<|channel|>commentary json<|message|>{"path":"tests/test_client.py"}'
    )

    assert benchmark._action_value(_Tokenizer(), text) == {
        "action": "run_tests",
        "path": "tests/test_client.py",
    }


def test_gpt_oss_action_accepts_native_function_channel():
    text = (
        "<|channel|>analysis<|message|>Apply the fix.<|end|>"
        "<|start|>assistant<|channel|>functions.replace_text<|message|>"
        '{"path":"app/client.py","old":"return timeout or DEFAULT_TIMEOUT",'
        '"new":"return timeout if timeout is not None else DEFAULT_TIMEOUT"}'
    )

    assert benchmark._action_value(_Tokenizer(), text) == {
        "action": "replace_text",
        "path": "app/client.py",
        "old": "return timeout or DEFAULT_TIMEOUT",
        "new": "return timeout if timeout is not None else DEFAULT_TIMEOUT",
    }


def test_gpt_oss_final_accepts_constrained_message():
    text = "<|channel|>final <|constrain|>**<|message|>All tests passed."

    assert benchmark._action_value(_Tokenizer(), text) == {
        "action": "finish",
        "summary": "All tests passed.",
    }
    assert benchmark._assistant_history(text) == {
        "role": "assistant",
        "content": "All tests passed.",
    }


def test_gpt_oss_final_accepts_direct_constrained_json():
    text = (
        "<|channel|>analysis<|message|>We need to finish.<|end|>"
        "<|start|>assistant<|channel|>final <|constrain|>"
        '{"summary":"All tests pass. No further changes needed."}'
    )

    assert benchmark._action_value(_Tokenizer(), text) == {
        "action": "finish",
        "summary": "All tests pass. No further changes needed.",
    }


def test_memory_guard_accepts_healthy_first_reading(monkeypatch):
    readings = iter((20,))
    monkeypatch.setattr(benchmark, "_free_percent", lambda: next(readings))

    result = benchmark._memory_guard(
        15, attempts=3, interval=0, reclaim=False
    )

    assert result == {"safe": True, "readings": [20], "reclaimed": False}


def test_memory_guard_recovers_after_cache_reclaim(monkeypatch):
    readings = iter((12, 16))
    calls = []
    monkeypatch.setattr(benchmark, "_free_percent", lambda: next(readings))
    monkeypatch.setattr(benchmark.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(benchmark.mx, "clear_cache", lambda: calls.append("mlx"))

    result = benchmark._memory_guard(
        15, attempts=3, interval=0, reclaim=True
    )

    assert result == {"safe": True, "readings": [12, 16], "reclaimed": True}
    assert calls == ["gc", "mlx"]


def test_memory_guard_requires_sustained_low_readings(monkeypatch):
    readings = iter((12, 13, 14))
    monkeypatch.setattr(benchmark, "_free_percent", lambda: next(readings))
    monkeypatch.setattr(benchmark.gc, "collect", lambda: None)
    monkeypatch.setattr(benchmark.mx, "clear_cache", lambda: None)

    result = benchmark._memory_guard(
        15, attempts=3, interval=0, reclaim=True
    )

    assert result == {
        "safe": False,
        "readings": [12, 13, 14],
        "reclaimed": True,
    }
