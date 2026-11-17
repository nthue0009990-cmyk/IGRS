"""Retiming neighborhood of a complete incumbent on the 34-node feeder."""
from .nature34_clpu_search import graph_plan


def improving_retimings(env, actions):
    """Shorten one wait block, advancing its whole suffix and padding the end."""
    actions = tuple(actions)
    base = graph_plan(env, env.initial, actions)
    best_energy = base['energy_kwh']
    output = []
    for end, action in enumerate(actions):
        if not action or not end or actions[end - 1] != 0:
            continue
        start = end - 1
        while start > 0 and actions[start - 1] == 0:
            start -= 1
        for advance in range(1, end - start + 1):
            plan = actions[:end - advance] + actions[end:] + (0,) * advance
            # Move only waits. Starting commands, non-wait order and final
            # configuration are fixed. Replay still checks the full graph path.
            info = graph_plan(env, env.initial, plan)
            assert info['states'][-1].command == base['states'][-1].command
            if info['energy_kwh'] > best_energy:
                output.append({'actions': list(plan), 'energy_kwh': info['energy_kwh'],
                               'advance_slots': advance, 'suffix_start_slot': end + 1})
    return output


def choose_retiming(pending, tried_energies):
    """Try the largest gain, then the smallest, then spread remaining probes."""
    def key(row):
        energy = row['energy_kwh']
        if not tried_energies:
            score = energy
        elif len(tried_energies) == 1:
            score = -energy
        else:
            score = min(abs(energy - previous) for previous in tried_energies)
        return (-score, -energy, row['actions'])
    return min(pending, key=key)
