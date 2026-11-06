"""The prediction-store read gate (memory limit on Render, settings.DATA_READ_CONCURRENCY)."""

from __future__ import annotations

import threading
import time

from app.data.store import _ReadGate


def test_gate_caps_parallel_readers() -> None:
    gate = _ReadGate(2)
    lock = threading.Lock()
    active = 0
    peak = 0

    def read() -> None:
        nonlocal active, peak
        with gate.hold():
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1

    threads = [threading.Thread(target=read) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 2


def test_gate_is_reentrant_in_one_thread() -> None:
    gate = _ReadGate(1)
    done = threading.Event()

    def nested() -> None:
        with gate.hold():
            with gate.hold():
                done.set()

    t = threading.Thread(target=nested)
    t.start()
    t.join(timeout=2)
    assert done.is_set()
    # The slot is free again afterwards.
    with gate.hold():
        pass
