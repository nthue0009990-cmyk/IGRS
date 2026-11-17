"""Restricted Yuan–Illindala–Khalsa tie-closure trellis, with a QSTS oracle.

Source: IEEE Transactions on Power Delivery 32(1), 310–319, Secs. II.B
and III.B/Fig. 5. This is a reimplementation of the no-loop branch. The
source ranks static switching states. Recording the prefix, placing one
operation per minute and checking the remaining horizon are common-study
extensions. A survivor per tie-set is a heuristic under CLPU.
"""
from math import isfinite, isclose


def _state_cost(row):
    return (-row["min_voltage_pu"], row["loss_kw"], row["prefix"])


def _solution_key(row):
    return (-row["terminal_kw"], len(row["prefix"]), *_state_cost(row))


def solve_prefix(extensions, evaluate_prefix, target_load_kw, horizon=30):
    """Published-rule state selection using operated slots only.

    evaluate_prefix returns PASS or reliable FAIL on the prefix's elapsed
    slots. No waiting tail or rival plans may inform voltage/loss ranking.
    The caller independently checks the chosen horizon-padded sequence and
    records failure separately from any protected output.
    """
    if horizon < 1 or not isfinite(target_load_kw) or target_load_kw <= 0:
        raise ValueError("Invalid horizon or total feeder demand")

    def score(prefix):
        row = dict(evaluate_prefix(prefix), prefix=prefix)
        if row.get("status") not in {"PASS", "FAIL"}:
            raise ValueError("Unreliable or budget-limited prefix passed to the selector")
        if row["status"] == "PASS" and (
                not isfinite(row["terminal_kw"]) or row["terminal_kw"] < 0
                or not isfinite(row["min_voltage_pu"]) or not isfinite(row["loss_kw"])):
            raise ValueError("Missing published-rule ranking metrics")
        return row

    best = score(())
    if best["status"] != "PASS":
        raise ValueError("Initial state has no valid physical certificate")
    initial = best
    parents = {frozenset(): initial}
    records = []
    stages = []
    full = lambda row: isclose(row["terminal_kw"], target_load_kw, rel_tol=1e-12, abs_tol=1e-10)
    stop = "ALL_LOAD_RESTORED" if full(initial) else "LEGAL_FEASIBLE_TRELLIS_EXHAUSTED"
    for depth in range(1, min(4, horizon) + 1):
        if full(best):
            break
        survivors = {}
        layer = []
        for previous, parent in sorted(parents.items(), key=lambda item: item[1]["prefix"]):
            actions = list(extensions(parent["prefix"]))
            if len(actions) != len(set(actions)) or any(
                    type(a) is not int or a not in range(16, 20) or a in previous for a in actions):
                raise ValueError("Invalid no-loop tie-only transition")
            for action in sorted(actions):
                row = score(parent["prefix"] + (action,))
                layer.append(row)
                if row["status"] != "PASS":
                    continue
                state = previous | {action}
                if state not in survivors or _state_cost(row) < _state_cost(survivors[state]):
                    survivors[state] = row
        records.extend(survivors.values())
        stages.append({"depth": depth, "evaluated": layer,
                       "survivor_prefixes": [r["prefix"] for r in sorted(survivors.values(), key=_state_cost)]})
        if not survivors:
            break
        parents = survivors
        best = min([best, *survivors.values()], key=_solution_key)
        if full(best):
            stop = "ALL_LOAD_RESTORED"
            break
    return {"prefix":list(best["prefix"]), "actions":list(best["prefix"])+[0]*(horizon-len(best["prefix"])),
            "selected":best, "stages":stages, "initial":initial, "stop_reason":stop,
            "ranked_prefixes":[r["prefix"] for r in sorted([initial,*records], key=_solution_key)]}


def _check(prefix, evaluate):
    data = dict(evaluate(prefix))
    if data.get("status") not in {"COMPLETE", "FAIL"}:
        raise ValueError("Expected a certified complete or reliable failed trajectory")
    if type(data.get("prefix_feasible")) is not bool:
        raise ValueError("Need the feasibility of the operated slots separately from the waiting tail")
    if not isfinite(data["terminal_kw"]) or data["terminal_kw"] < 0:
        raise ValueError("Invalid connected normal load")
    if data["prefix_feasible"]:
        if not isfinite(data["min_voltage_pu"]) or not isfinite(data["loss_kw"]):
            raise ValueError("A feasible switching state requires its paid stage-end ranking metrics")
    if data["status"] == "COMPLETE":
        if not data["prefix_feasible"] or data.get("energy_kwh") is None or not isfinite(data["energy_kwh"]):
            raise ValueError("Cannot select an incomplete trajectory")
    elif data.get("energy_kwh") is not None:
        raise ValueError("A failed trajectory cannot carry a certified objective")
    return {**data, "prefix": tuple(prefix)}


def solve(extensions, evaluate, target_load_kw, horizon=30):
    """Return the selected complete schedule on the explicitly limited alphabet.

    extensions(p) lists feasible graph transitions only; every potential
    electrical transition calls evaluate. An electrical rejection in the
    WAIT tail of a padded candidate must not invalidate its operated prefix.
    Selection of a published-rule state does not transfer any AC evidence
    between different switching histories. Exact ranking ties use action IDs
    rather than the random tie break in the original paper.
    """
    if horizon < 1 or not isfinite(target_load_kw) or target_load_kw <= 0:
        raise ValueError("Invalid horizon or total feeder demand")
    initial = _check((), evaluate)
    if initial["status"] != "COMPLETE":
        raise ValueError("The common all-wait fallback requires a complete certificate")
    best = initial
    parents = {frozenset(): initial}
    stages = []
    full = lambda row: isclose(row["terminal_kw"], target_load_kw, rel_tol=1e-12, abs_tol=1e-10)
    stop = "ALL_LOAD_RESTORED" if full(initial) else "NO_REMAINING_LEGAL_FEASIBLE_PREFIX"
    for depth in range(1, min(4, horizon) + 1):
        if full(best):
            break
        survivors = {}
        layer_complete = []
        evaluated = []
        for previous, parent in sorted(parents.items(), key=lambda item: item[1]["prefix"]):
            actions = list(extensions(parent["prefix"]))
            if any(type(a) is not int or a not in range(16, 20) or a in previous for a in actions):
                raise ValueError("The restricted problem supports one closure of each normally open tie")
            if len(actions) != len(set(actions)):
                raise ValueError("Duplicate transition for the same parent")
            for action in sorted(actions):
                prefix = parent["prefix"] + (action,)
                row = _check(prefix, evaluate)
                evaluated.append(row)
                if row["status"] == "COMPLETE":
                    layer_complete.append(row)
                if row["prefix_feasible"]:
                    state = previous | {action}
                    if state not in survivors or _state_cost(row) < _state_cost(survivors[state]):
                        survivors[state] = row
        stages.append({"depth": depth, "evaluated": evaluated,
                       "survivor_prefixes": [r["prefix"] for r in sorted(survivors.values(), key=_state_cost)]})
        if layer_complete:
            best = min([best, *layer_complete], key=_solution_key)
        if full(best):
            stop = "ALL_LOAD_RESTORED"
            break
        if not survivors:
            break
        parents = survivors
    return {"actions": list(best["prefix"]) + [0] * (horizon - len(best["prefix"])),
            "prefix": list(best["prefix"]), "selected": best, "stages": stages,
            "initial": initial, "stop_reason": stop}
