"""Two-stage target/sequence interface checks, with no physical solver.

This module checks the adapted formulation's restrictions in addition to the
existing complete-plan graph checks. It never returns a feasibility certificate
or supplies a terminal optimizer in place of the electrical MILP.
"""
from .nature34_clpu_search import graph_plan


VERSION = 'nc34-two-stage-static-plan-contract-v1'


def check_target_plan(env, actions, *, target_switch_mask, required_load_names):
    """Check an initial-to-terminal 30-slot plan; required loads are a subset.

    Equal initial/target line bits must remain fixed. Other line bits can move
    once toward their target. Readiness bits follow the common environment's
    actual-service monotonicity, so preparation and subsequent pickup remain
    possible. These are graph/interface checks, not an electrical validation.
    """
    if (env.horizon, env.slot_seconds, env.sample_seconds) != (30, 60, 30):
        raise ValueError('The adapted contract requires the frozen 30-slot grid')
    if type(target_switch_mask) is not int or not 0 <= target_switch_mask < 512:
        raise ValueError('Target must contain exactly nine integer line bits')
    if not isinstance(required_load_names, (list, tuple, set, frozenset)):
        raise ValueError('Required loads must be a collection of exact names')
    if any(type(n) is not str for n in required_load_names):
        raise ValueError('Required load names must be strings')
    required = frozenset(required_load_names)
    if len(required) != len(required_load_names) or not required <= set(env.load_names):
        raise ValueError('Unknown or duplicated required load')
    if not env.initial_energized <= required:
        raise ValueError('Terminal target cannot exclude initially supplied loads')
    info = graph_plan(env, env.initial, actions)
    states = info['states']
    initial_mask = env.initial.command >> 10
    allowed_changes = initial_mask ^ target_switch_mask
    previous_changes = 0
    for state in states[1:]:
        changes = (state.command >> 10) ^ initial_mask
        if changes & ~allowed_changes or previous_changes & ~changes:
            raise ValueError('Line sequence departs from the one-way target transitions')
        previous_changes = changes
    terminal_mask = states[-1].command >> 10
    if terminal_mask != target_switch_mask:
        raise ValueError('Terminal line topology does not match the target')
    served = env.energized(states[-1].command)
    if not required <= served:
        raise ValueError('Some required terminal loads remain unserved')
    return {
        'version': VERSION,
        'status': 'GRAPH_VALID_UNCERTIFIED',
        'actions': list(info['actions']),
        'target_switch_mask': target_switch_mask,
        'required_load_names': sorted(required),
        'terminal_served_load_names': sorted(served),
        'additional_served_load_names': sorted(served - required),
        'unserved_load_names': sorted(set(env.load_names) - served),
        'states': [s.to_dict() for s in states],
        'pickup_times_seconds': dict(zip(env.load_names, states[-1].pickups)),
        'graph_energy_kwh': info['energy_kwh'],
        'operations': info['operations'],
        'priority': info['priority'],
        'certified_energy_kwh': None,
        'physical_validation': None,
        'new_physical_points': 0,
    }
