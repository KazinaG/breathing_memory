from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import json
from pathlib import Path
import threading
import time
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


class AnnMaintenanceCoordinator:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        suffix = self.db_path.suffix or ".sqlite3"
        self.lock_path = self.db_path.with_suffix(f"{suffix}.ann.lock")
        self.status_path = self.db_path.with_suffix(f"{suffix}.ann.status.json")
        self._process_lock = threading.RLock()
        self._thread_state = threading.local()

    @contextmanager
    def acquire_shared(self, timeout_ms: int | None = None) -> Iterator[bool]:
        lease = self._acquire(shared=True, timeout_ms=timeout_ms)
        if lease is None:
            yield False
            return
        try:
            yield True
        finally:
            lease.release()

    @contextmanager
    def acquire_exclusive(self, timeout_ms: int | None = None) -> Iterator[bool]:
        lease = self._acquire(shared=False, timeout_ms=timeout_ms)
        if lease is None:
            yield False
            return
        try:
            yield True
        finally:
            lease.release()

    @contextmanager
    def active_status(
        self,
        *,
        code: str,
        phase: str,
        fragment_count: int,
        current_mode: str,
        suggested_action: str,
    ) -> Iterator[None]:
        payload = {
            "code": code,
            "retryable": True,
            "phase": phase,
            "fragment_count": int(fragment_count),
            "current_mode": current_mode,
            "suggested_action": suggested_action,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "started_epoch_ms": int(time.time() * 1000),
        }
        self._write_status(payload)
        try:
            yield
        finally:
            self._clear_status()

    def read_status(self, *, current_mode: str) -> dict:
        payload = self._read_status_payload()
        if payload is None:
            return {
                "code": "rebuild_in_progress",
                "retryable": True,
                "phase": "ann_maintenance",
                "current_mode": current_mode,
                "suggested_action": "retry",
            }
        payload = dict(payload)
        started_epoch_ms = payload.pop("started_epoch_ms", None)
        if isinstance(started_epoch_ms, int):
            payload["elapsed_ms"] = max(0, int(time.time() * 1000) - started_epoch_ms)
        payload.setdefault("current_mode", current_mode)
        payload.setdefault("suggested_action", "retry")
        return payload

    def _acquire(self, *, shared: bool, timeout_ms: int | None):
        depth = getattr(self._thread_state, "depth", 0)
        mode = getattr(self._thread_state, "mode", None)
        if depth > 0:
            if mode == "exclusive" or (mode == "shared" and shared):
                self._thread_state.depth = depth + 1
                return _NestedLease(self)
            raise RuntimeError("cannot upgrade a shared maintenance lock to exclusive")

        deadline = None if timeout_ms is None else time.monotonic() + (timeout_ms / 1000.0)
        if not self._acquire_process_lock(deadline):
            return None

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = self.lock_path.open("a+b")
        if not self._acquire_file_lock(fd, shared=shared, deadline=deadline):
            fd.close()
            self._process_lock.release()
            return None

        self._thread_state.depth = 1
        self._thread_state.mode = "shared" if shared else "exclusive"
        self._thread_state.fd = fd
        return _Lease(self)

    def _acquire_process_lock(self, deadline: float | None) -> bool:
        if deadline is None:
            self._process_lock.acquire()
            return True
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if self._process_lock.acquire(timeout=min(remaining, 0.05)):
                return True

    def _acquire_file_lock(self, fd, *, shared: bool, deadline: float | None) -> bool:
        if fcntl is None:  # pragma: no cover
            return True
        operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        while True:
            try:
                fcntl.flock(fd.fileno(), operation | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                if deadline is not None and time.monotonic() >= deadline:
                    return False
                time.sleep(0.01)

    def _release(self) -> None:
        depth = getattr(self._thread_state, "depth", 0)
        if depth <= 0:
            return
        if depth > 1:
            self._thread_state.depth = depth - 1
            return
        fd = self._thread_state.fd
        if fcntl is not None:  # pragma: no branch
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        fd.close()
        self._thread_state.depth = 0
        self._thread_state.mode = None
        self._thread_state.fd = None
        self._process_lock.release()

    def _write_status(self, payload: dict) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _clear_status(self) -> None:
        if self.status_path.exists():
            self.status_path.unlink()

    def _read_status_payload(self) -> dict | None:
        if not self.status_path.exists():
            return None
        try:
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except Exception:
            return None


class _Lease:
    def __init__(self, coordinator: AnnMaintenanceCoordinator):
        self.coordinator = coordinator

    def release(self) -> None:
        self.coordinator._release()


class _NestedLease:
    def __init__(self, coordinator: AnnMaintenanceCoordinator):
        self.coordinator = coordinator

    def release(self) -> None:
        self.coordinator._release()
