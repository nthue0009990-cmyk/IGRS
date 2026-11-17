"""Read the archived Hachmann 2022 CLPU events without refitting or filtering.

The source parameters.csv uses decimal commas *and* field commas.  Its four
nonzero decimal fields occupy token pairs; Td is either the literal zero or a
fifth pair.  We accept that observed schema only, and independently compare ED
and DED equations to every supplied fitted point.  Thus an ambiguous CSV parse
cannot silently become an experimental parameter.

This module supplies post-reconnection curves, not an outage-age model.  The
consumer owns actual reconnection timestamps and the off/never-interrupted
states.  A disconnected load must not decay before it is first reconnected.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
from typing import Mapping, Sequence
import zipfile


SOURCE_ZIP_SHA256 = "e46886903929d49380a7f1342588adebd26c2524667dbd74fa89021ccea27a22"
DEFAULT_SPLIT_DOMAIN = "clpu-event-split-v1"
SOURCE_EVENT_COUNT = 31
# Four decimal places in the archived power series: half one printed unit.
PRINTED_POWER_HALF_UNIT = 0.00005
NUMERICAL_ROUNDOFF_ALLOWANCE = 1e-12
_KINDS = ("raw_", "fit_", "fit3_", "step_to_max", "step_to_steadystate")
_INTEGER = re.compile(r"[+-]?\d+\Z")
_DECIMAL_TAIL = re.compile(r"\d+(?:[Ee][+-]?\d+)?\Z")
_SERIES_ROW = re.compile(r"([+-]?\d+\.\d{3}),\s*([+-]?\d+\.\d{4})\Z")


class CurveDataError(ValueError):
    """The archived event data cannot be used with the declared schema."""


def _decimal_pair(integer: str, fractional: str) -> float:
    if not _INTEGER.fullmatch(integer) or not _DECIMAL_TAIL.fullmatch(fractional):
        raise CurveDataError(f"Invalid decimal-comma pair: {integer!r}, {fractional!r}")
    value = float(integer + "." + fractional)
    if not math.isfinite(value):
        raise CurveDataError("Nonfinite parameter")
    return value


def _parse_parameter_row(row: str, event_id: int) -> dict:
    tokens = [part.strip() for part in row.split(",")]
    if len(tokens) not in (9, 10):
        raise CurveDataError(f"Event {event_id}: expected 9 or 10 comma tokens")
    values = [_decimal_pair(tokens[j], tokens[j + 1]) for j in range(0, 8, 2)]
    if len(tokens) == 9:
        if tokens[8] != "0":
            raise CurveDataError(f"Event {event_id}: unpaired Td must be literal zero")
        delay = 0.0
    else:
        delay = _decimal_pair(tokens[8], tokens[9])
    a, tau, ded_a, ded_tau = values
    if a < 0 or ded_a < 0 or tau <= 0 or ded_tau <= 0 or delay < 0:
        raise CurveDataError(f"Event {event_id}: invalid ED/DED parameter domain")
    return {
        "event_id": event_id,
        "a": a,
        "tau_seconds": tau,
        "ded_a": ded_a,
        "ded_tau_seconds": ded_tau,
        "ded_delay_seconds": delay,
        "parameter_row": row,
    }


def multiplier(event: Mapping, age_seconds: float) -> float:
    """Return 1 + a exp(-age/tau) at a finite post-reconnection age >= 0.

    At age zero this is the full 1+a peak regardless of how long the load waited
    while off.  Negative/unknown ages are errors, not a license to pre-decay.
    """
    age = float(age_seconds)
    a, tau = float(event["a"]), float(event["tau_seconds"])
    if not all(math.isfinite(value) for value in (age, a, tau)):
        raise ValueError("CLPU age and parameters must be finite")
    if age < 0 or a < 0 or tau <= 0:
        raise ValueError("CLPU requires age >= 0, a >= 0, and tau > 0")
    return 1.0 + a * math.exp(-age / tau)


def _series(blob: bytes, member: str) -> list[tuple[float, float]]:
    lines = [line.strip() for line in blob.decode("utf-8-sig").splitlines() if line.strip()]
    if not lines or lines[0] != "1":
        raise CurveDataError(f"{member}: expected one-channel header")
    points = []
    for line in lines[1:]:
        match = _SERIES_ROW.fullmatch(line)
        if match is None:
            raise CurveDataError(f"{member}: unexpected series precision/schema")
        time, power = map(float, match.groups())
        if not math.isfinite(time) or not math.isfinite(power) or time < 0:
            raise CurveDataError(f"{member}: invalid series values")
        if points and time <= points[-1][0]:
            raise CurveDataError(f"{member}: non-increasing timestamps")
        points.append((time, power))
    if not points:
        raise CurveDataError(f"{member}: empty time series")
    return points


def load_events(zip_path: str | Path) -> list[dict]:
    """Load all 31 paired ED events, validating all ten series per event.

    Returns a fresh list ordered by event_id.  Each event includes paired DED
    metadata, source hashes and formula/shift crosschecks.  No raw ZIP bytes are
    changed, and no sample is removed for a tiny amplitude or extreme tau.
    Invalid or missing evidence raises CurveDataError before returning events.
    """
    zip_path = Path(zip_path)
    source_hash = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    events: list[dict] = []
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise CurveDataError("Duplicate member names in source ZIP")
        try:
            parameter_blob = archive.read("parameters.csv")
            explanation_blob = archive.read("Explaination.txt")
        except KeyError as error:
            raise CurveDataError("Missing source parameter table/explanation") from error
        rows = [row.strip() for row in parameter_blob.decode("utf-8-sig").splitlines() if row.strip()]
        if not rows or not rows[0].startswith("ED model, parameter a in p.u.,ED model, parameter Tau in s,"):
            raise CurveDataError("Unexpected source parameter header")
        if len(rows) - 1 != SOURCE_EVENT_COUNT:
            raise CurveDataError("Expected exactly 31 complete reconnection events")
        for event_id, row in enumerate(rows[1:], 1):
            event = _parse_parameter_row(row, event_id)
            series_hashes = {}
            counts = {}
            maxima = {"ed": 0.0, "ded": 0.0, "step_max": 0.0, "step_steady": 0.0}
            normalized = {}
            for connection_time in (300, 600):
                for kind in _KINDS:
                    member = f"TimeseriesConnectionAt{connection_time}s/{kind}{event_id}.txt"
                    try:
                        blob = archive.read(member)
                    except KeyError as error:
                        raise CurveDataError(f"Missing complete-event member {member}") from error
                    points = _series(blob, member)
                    before = [(t, y) for t, y in points if t < connection_time]
                    post = [(t - connection_time, y) for t, y in points if t >= connection_time]
                    if not before or any(y != 0.0 for _, y in before) or not post or post[0][0] != 0.0:
                        raise CurveDataError(f"{member}: invalid disconnected/reconnection boundary")
                    normalized[(connection_time, kind)] = post
                    counts[member] = len(post)
                    series_hashes[member] = hashlib.sha256(blob).hexdigest()
                    for age, power in post:
                        if kind == "raw_":
                            continue  # A measured trace need not match its fitted model.
                        if kind == "fit_":
                            expected, model = multiplier(event, age), "ed"
                        elif kind == "fit3_":
                            ded_age = max(0.0, age - event["ded_delay_seconds"])
                            expected = 1.0 + event["ded_a"] * math.exp(-ded_age / event["ded_tau_seconds"])
                            model = "ded"
                        elif kind == "step_to_max":
                            expected, model = 1.0 + event["a"], "step_max"
                        else:
                            expected, model = 1.0, "step_steady"
                        error = abs(power - expected)
                        maxima[model] = max(maxima[model], error)
                        if error > PRINTED_POWER_HALF_UNIT + NUMERICAL_ROUNDOFF_ALLOWANCE:
                            raise CurveDataError(f"{member}: {model} mismatch at age {age}: {error}")
            for kind in _KINDS:
                if normalized[(300, kind)] != normalized[(600, kind)]:
                    raise CurveDataError(f"Event {event_id}: 300/600 s {kind} versions are not identical shifts")
            event["source_zip_sha256"] = source_hash
            event["validation"] = {
                "passed": True,
                "shifted_versions_identical": True,
                "post_reconnection_points": counts,
                "formula_max_abs_error": maxima,
                "printed_power_half_unit": PRINTED_POWER_HALF_UNIT,
                "numerical_roundoff_allowance": NUMERICAL_ROUNDOFF_ALLOWANCE,
                "series_sha256": series_hashes,
                "parameters_sha256": hashlib.sha256(parameter_blob).hexdigest(),
                "explanation_sha256": hashlib.sha256(explanation_blob).hexdigest(),
            }
            events.append(event)
    return events


def split_events(events: Sequence[Mapping], domain: str = DEFAULT_SPLIT_DOMAIN) -> dict[str, list[dict]]:
    """Freeze a 19/6/6 split by sha256(domain + '|' + decimal event_id).

    Entire events, including both shifted files and all fitted/raw variants,
    follow one partition.  Parameters and outcomes have no role in sorting.
    """
    ids = [event["event_id"] for event in events]
    if len(ids) != SOURCE_EVENT_COUNT or set(ids) != set(range(1, SOURCE_EVENT_COUNT + 1)):
        raise ValueError("Splitting requires all 31 unique event IDs")
    if not isinstance(domain, str) or not domain:
        raise ValueError("Split domain must be a nonempty string")
    ordered = sorted(events, key=lambda event: hashlib.sha256(f"{domain}|{event['event_id']}".encode("utf-8")).hexdigest())
    return {
        "train": [dict(event) for event in ordered[:19]],
        "dev": [dict(event) for event in ordered[19:25]],
        "holdout": [dict(event) for event in ordered[25:]],
    }
