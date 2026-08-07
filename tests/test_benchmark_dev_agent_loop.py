from benchmarks import benchmark_dev_agent_loop as benchmark


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
