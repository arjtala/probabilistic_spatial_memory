"""Throttled progress + stage logging for long-running extractions.

Two minimal helpers:
    - stage_banner(stage, msg) prints a one-line stage transition to stderr.
    - make_progress_logger(stage, n_total) returns a callback the runner
      invokes after each batch with the running count. Output is throttled
      so a long inference loop doesn't flood the terminal — first/last
      points always print, middle ones are rate-limited.

stderr (not stdout) so the JSON manifest still pipes cleanly through jq.
"""

import sys
import time
from collections.abc import Callable


def _fmt_time(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def stage_banner(stage: str, msg: str = "", *, stream=sys.stderr) -> None:
    if msg:
        print(f"[{stage}] {msg}", file=stream, flush=True)
    else:
        print(f"[{stage}]", file=stream, flush=True)


def make_progress_logger(
    stage: str,
    n_total: int,
    *,
    throttle_sec: float = 2.0,
    stream=sys.stderr,
) -> Callable[[int], None]:
    """Return a `cb(n_done)` callable that emits throttled progress lines.

    The first call always prints (so the user sees "1/N" immediately rather
    than waiting throttle_sec to learn the run is alive). The final call
    (n_done >= n_total) also always prints to mark completion. Middle calls
    print at most every `throttle_sec` seconds.
    """
    state = {"last_t": 0.0, "start_t": time.time(), "first": True}

    def cb(n_done: int) -> None:
        now = time.time()
        is_final = n_done >= n_total
        if not state["first"] and not is_final and now - state["last_t"] < throttle_sec:
            return
        state["first"] = False
        state["last_t"] = now
        elapsed = now - state["start_t"]
        rate = n_done / elapsed if elapsed > 0 else 0.0
        eta = (n_total - n_done) / rate if rate > 0 else 0.0
        pct = (100.0 * n_done / n_total) if n_total > 0 else 0.0
        print(
            f"[{stage}] {n_done}/{n_total} ({pct:5.1f}%) "
            f"{rate:6.1f} it/s  elapsed={_fmt_time(elapsed)}  eta={_fmt_time(eta)}",
            file=stream,
            flush=True,
        )

    return cb
