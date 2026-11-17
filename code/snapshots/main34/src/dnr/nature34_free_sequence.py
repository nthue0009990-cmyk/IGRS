"""Complete switch assignments with fixed all-ready load commands for NC34.

This is a free-terminal, reversible switch graph, not the full joint switch and
load-readiness domain. Every nine-bit assignment is retained. Faulted closed
switch assignments and nonforest topologies remain explicit records but receive
no admissible edges. Forest status alone does not certify electrical feasibility.
"""
from __future__ import annotations

from collections.abc import Mapping

from dnr.nature34_sequence_catalog import initial_command, state_metadata


def enumerate_switch_space(static: Mapping, loads: Mapping, scenario: Mapping) -> dict:
    """Enumerate all 512 raw assignments; graph edges flip one healthy switch.

    Mask bit ``i`` is switch coordinate ``i``. Ten readiness commands always stay
    one, including at unpowered buses. Effective load bits are derived from
    reachability. Invalid fault-closed assignments are not silently remapped to
    another command or counted as evaluated topologies. The caller separately
    certifies forest states at each condition and fixed operating point.
    """
    initial = initial_command(static, scenario)  # Validate static graph and faults.
    initial_meta = state_metadata(static, loads, scenario, initial)
    original = {str(k).lower(): frozenset(str(x).lower() for x in edge)
                for k, edge in static["original_lines"].items()}
    fault_edges = {original[str(name).lower()] for name in scenario["fault_lines"]}
    forced = [i for i, edge in enumerate(static["switch_edges"])
              if frozenset(str(x).lower() for x in edge) in fault_edges]
    states = {}
    for mask in range(512):
        state_id = f"s{mask:03x}"
        command = [(mask >> i) & 1 for i in range(9)] + [1] * 10
        if any(command[i] for i in forced):
            metadata = {"command": command, "effective_action": None, "forest": None,
                        "source_reachable_buses": None, "deenergized_buses": None,
                        "energized_load_ids": None, "deenergized_load_ids": None,
                        "deliberately_shed_load_ids": None, "base_served_kw": None,
                        "base_denominator_kw": initial_meta["base_denominator_kw"],
                        "physical_feasibility_inferred": False}
            status = "FAULTED_SWITCH_CLOSED"
        else:
            metadata = state_metadata(static, loads, scenario, command)
            status = "FOREST_TOPOLOGY_ONLY" if metadata["forest"] else "NONFOREST"
        states[state_id] = {"state_id": state_id, "mask": mask, **metadata,
                            "command_bits": command.copy(), "topology_status": status}
    successors = {}
    for sid, row in states.items():
        successors[sid] = sorted(f"s{row['mask'] ^ (1 << i):03x}" for i in range(9)
                                 if row["forest"] is True and
                                 states[f"s{row['mask'] ^ (1 << i):03x}"]["forest"] is True)
    arcs = sum(map(len, successors.values()))
    initial_mask = sum(bit << i for i, bit in enumerate(initial[:9]))
    return {"schema_version": 1, "scenario_id": scenario.get("scenario_id"),
            "initial_id": f"s{initial_mask:03x}", "initial_command": initial,
            "states": states, "successors": successors,
            "counts": {"switch_assignments": 512,
                       "forest_states": sum(r["forest"] is True for r in states.values()),
                       "nonforest_states": sum(r["forest"] is False for r in states.values()),
                       "fault_closed_assignments": sum(r["forest"] is None for r in states.values()),
                       "directed_forest_edges": arcs, "undirected_forest_edges": arcs // 2},
            "forced_open_switch_indices": forced,
            "scope": "complete_nine_switch_assignments_fixed_all_ready_free_terminal",
            "load_readiness_changes_allowed": False, "physical_feasibility_inferred": False,
            "new_physical_solves": 0,
            "stay_rule": "sequence solver adds stay at every physically certified state",
            "operating_point_rule": "caller must build a separate graph for each fixed alpha and condition"}
