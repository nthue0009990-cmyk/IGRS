"""Append-and-flush call evidence for the named NC34 DSS adapter, with no solves.

Requests reserve documented command entries; only a returned API call closes a
RETURNED record. Neither state proves convergence, physical validity, internal
iteration counts, or exact execution of entries within an interrupted compile.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading

from .dss_command_inventory import inventory_dss_commands, _literal_include, _without_line_comment

COUNT_KEYS = ("python_solve_commands", "script_solve_commands", "text_solve_commands", "compile_commands",
              "explicit_solve_commands", "calcvoltagebases_entries", "numerical_solution_entries")


def _zero_counts():
    return {key: 0 for key in COUNT_KEYS}


def _json_ready(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return {"type": type(value).__name__, "repr": repr(value)}


def _text_value(args, kwargs):
    if args:
        return args[0]
    values = [kwargs[key] for key in ("value", "command", "cmd") if key in kwargs]
    return values[0] if len(values) == 1 else None


class DurableCounterFactory:
    """Callable replacement for EngineCounter(target, events, prefix='').

    stats() separately reports requested/returned/unknown_completion/not_invoked
    entry budgets. A successful new point should have returned delta(3 explicit,
    1 CalcVoltageBases), no failed/rejected calls and no open call IDs.
    """
    def __init__(self, ledger_path, *, expected_compile_inventory, context_provider=None, fsync=False):
        expected = expected_compile_inventory
        if expected.get("explicit_solve_commands") != 1 or expected.get("calcvoltagebases_entries") != 1:
            raise ValueError("Named compile inventory must contain exactly one Solve and one CalcVoltageBases")
        self._expected_files = {key: value["sha256"] for key, value in expected["files"].items()}
        self._expected_entry = Path(expected["entry_file"]).name
        if not self._expected_files:
            raise ValueError("Named compile inventory must preserve source SHA evidence")
        self.ledger_path = Path(ledger_path).resolve()
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        if self.ledger_path.exists() and self.ledger_path.stat().st_size:
            raise ValueError("Use a fresh ledger; an existing nonempty ledger must not be reused or overwritten")
        self._stream = self.ledger_path.open("a", encoding="utf-8", newline="\n")
        self._context_provider = context_provider or (lambda: {})
        self._fsync = fsync
        self._lock = threading.RLock()
        self._next_call_id = 0
        self._records = 0
        self._closed = False
        self._open = {}
        self._totals = {name: _zero_counts() for name in ("requested", "returned", "unknown_completion", "not_invoked")}
        self._call_totals = {name: 0 for name in ("calls_started", "calls_returned", "calls_failed", "calls_rejected")}

    def __call__(self, target, events, prefix=""):
        if self._closed:
            raise ValueError("Durable call ledger is closed")
        return DurableEngineCounter(target, events, prefix, self)

    def _append(self, record):
        encoded = json.dumps(_json_ready(record), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        self._stream.write(encoded + "\n")
        self._stream.flush()
        if self._fsync:
            os.fsync(self._stream.fileno())
        self._records += 1

    def _plan(self, method, args, kwargs):
        counts = _zero_counts()
        command = _text_value(args, kwargs) if method == "Text.Command" else None
        token = command.lstrip().split(None, 1)[0].lower() if isinstance(command, str) and command.strip() else ""
        if method == "Solution.Solve":
            counts["python_solve_commands"] = 1
        elif token == "compile":
            counts["compile_commands"] = counts["script_solve_commands"] = counts["calcvoltagebases_entries"] = 1
        elif token == "solve":
            counts["text_solve_commands"] = 1
        elif token == "calcvoltagebases":
            counts["calcvoltagebases_entries"] = 1
        counts["explicit_solve_commands"] = sum(counts[key] for key in ("python_solve_commands", "script_solve_commands", "text_solve_commands"))
        counts["numerical_solution_entries"] = counts["explicit_solve_commands"] + counts["calcvoltagebases_entries"]
        return counts, command, token

    def _validate_compile(self, command):
        code = _without_line_comment(command)
        parts = code.split(None, 1)
        if len(parts) != 2 or parts[0].lower() != "compile":
            raise ValueError("Malformed named Compile request")
        target = Path(_literal_include(parts[1])).resolve()
        actual = inventory_dss_commands(target)
        files = {key: value["sha256"] for key, value in actual["files"].items()}
        if target.name != self._expected_entry or files != self._expected_files or actual["explicit_solve_commands"] != 1 or actual["calcvoltagebases_entries"] != 1:
            raise ValueError("Actual compile source hashes/solution inventory differ from the frozen named source")
        return {"entry_file": str(target), "files": files, "explicit_solve_commands": 1, "calcvoltagebases_entries": 1,
                "validation": "MATCHED_FROZEN_STATIC_INVENTORY_BEFORE_API"}

    def _add(self, destination, counts):
        for key, value in counts.items():
            self._totals[destination][key] += value

    def _finish_failure(self, row, record, counts, error, invoked):
        status = "FAILED_UNKNOWN_COMPLETION" if invoked else "REJECTED_NOT_INVOKED"
        row.update(status=status, error=repr(error), exception_type=type(error).__name__, underlying_api_invoked=invoked)
        self._call_totals["calls_failed" if invoked else "calls_rejected"] += 1
        self._add("unknown_completion" if invoked else "not_invoked", counts)
        self._open.pop(record["call_id"], None)
        try:
            self._append({**record, "event": "call_failed" if invoked else "call_rejected", "status": status,
                          "closed_utc": datetime.now(timezone.utc).isoformat(), "error": repr(error),
                          "exception_type": type(error).__name__, "underlying_api_invoked": invoked,
                          "execution_completion": "UNKNOWN" if invoked else "NOT_INVOKED"})
        except BaseException as ledger_error:
            row["ledger_write_error"] = repr(ledger_error)
            error.add_note("Durable failure record could not be flushed: " + repr(ledger_error))

    def _call(self, target, events, method, args, kwargs):
        # The named physical adapter is serial; the lock also keeps paired IDs
        # and records coherent if a fake client invokes it from another thread.
        with self._lock:
            if self._closed:
                raise ValueError("Durable call ledger is closed")
            counts, command, token = self._plan(method, args, kwargs)
            call_id = self._next_call_id
            self._next_call_id += 1
            row = {"method": method, "args": list(args), "durable_call_id": call_id, "status": "REQUESTED"}
            if kwargs:
                row["kwargs"] = dict(kwargs)
            events.append(row)
            context_error = None
            try:
                context = _json_ready(self._context_provider())
            except BaseException as error:
                context, context_error = {"context_error": repr(error)}, error
            record = {"call_id": call_id, "method": method, "args": list(args), "kwargs": dict(kwargs),
                      "context": context, "requested_counts": counts, "requested_utc": datetime.now(timezone.utc).isoformat(),
                      "accounting_semantics": "REQUEST_PLAN_NOT_PROOF_OF_EXECUTION"}
            self._call_totals["calls_started"] += 1
            self._add("requested", counts)
            self._open[call_id] = counts
            try:
                self._append({**record, "event": "call_started", "status": "REQUESTED_NOT_YET_INVOKED",
                              "underlying_api_invoked": False, "execution_completion": "UNKNOWN"})
            except BaseException as error:
                row.update(status="REJECTED_START_LOG_FAILURE", error=repr(error), underlying_api_invoked=False)
                self._call_totals["calls_rejected"] += 1
                self._add("not_invoked", counts)
                self._open.pop(call_id, None)
                raise
            invoked = False
            try:
                if context_error is not None:
                    raise context_error
                if token == "compile":
                    record["compile_inventory"] = self._validate_compile(command)
                # The fixed adapter has no outer Redirect commands. An unbilled
                # external include must never silently enter its budget.
                if token == "redirect":
                    raise ValueError("Outer Redirect is outside this named counter scope; audit it before use")
                invoked = True
                result = target(*args, **kwargs)
            except BaseException as error:
                self._finish_failure(row, record, counts, error, invoked)
                raise
            row.update(status="RETURNED", underlying_api_invoked=True)
            try:
                self._append({**record, "event": "call_returned", "status": "RETURNED",
                              "returned_utc": datetime.now(timezone.utc).isoformat(), "underlying_api_invoked": True,
                              "execution_completion": "API_RETURNED_NOT_A_CONVERGENCE_OR_PHYSICS_CERTIFICATE"})
            except BaseException as error:
                row.update(status="API_RETURNED_COMPLETION_LOG_FAILED", error=repr(error))
                self._call_totals["calls_failed"] += 1
                self._add("unknown_completion", counts)
                self._open.pop(call_id, None)
                raise
            self._call_totals["calls_returned"] += 1
            self._add("returned", counts)
            self._open.pop(call_id, None)
            return result

    def stats(self):
        with self._lock:
            return {**self._call_totals, **{name: dict(values) for name, values in self._totals.items()},
                    "open_call_ids": sorted(self._open), "ledger_records": self._records,
                    "ledger_path": str(self.ledger_path), "flush_per_record": True, "fsync_per_record": self._fsync}

    def close(self):
        with self._lock:
            if not self._closed:
                self._stream.flush()
                if self._fsync:
                    os.fsync(self._stream.fileno())
                self._stream.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class DurableEngineCounter:
    """Transparent child proxies for Text/Solution; all other attributes pass through."""
    def __init__(self, target, events, prefix, factory):
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "prefix", prefix)
        object.__setattr__(self, "_factory", factory)

    def __getattr__(self, name):
        target = getattr(self.target, name)
        if name in ("Solution", "Text") and not self.prefix:
            return DurableEngineCounter(target, self.events, name, self._factory)
        if callable(target) and (self.prefix, name) in (("Solution", "Solve"), ("Text", "Command")):
            def call(*args, **kwargs):
                return self._factory._call(target, self.events, self.prefix + "." + name, args, kwargs)
            return call
        return target

    def __setattr__(self, name, value):
        if name in ("target", "events", "prefix", "_factory"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.target, name, value)


def make_durable_engine_counter(ledger_path, *, expected_compile_inventory, context_provider=None, fsync=False):
    """Create a callable EngineCounter replacement plus stats()/close() access."""
    return DurableCounterFactory(ledger_path, expected_compile_inventory=expected_compile_inventory,
                                 context_provider=context_provider, fsync=fsync)
