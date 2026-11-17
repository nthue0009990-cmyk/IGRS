"""Exact comparison across Python checkpoint and JSON record containers."""
import json


def identical_json_record(left, right):
    """Permit JSON's tuple-to-array transport, with no numeric tolerance.

    All fields and sequence order must match. Non-finite values are rejected.
    """
    options = dict(sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return json.dumps(left, **options) == json.dumps(right, **options)
