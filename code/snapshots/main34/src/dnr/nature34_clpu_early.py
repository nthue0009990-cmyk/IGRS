"""Frozen early-selector intervention and independent complete Beam guard.

This module neither constructs an oracle nor changes a search state. EARLY and
GUARD are separate cold searches; all physical and logical costs are summed by
the runner. Only EARLY turns 0..15 may use a fixed actor.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .nature34_clpu_environment import CLPUReliabilityError

VERSION = 'early16-window-independent-beam-guard-v1'
RESULT_SCHEMA = 'clpu-early-protected-pilot-result-v1'
EARLY_STEPS = 16


def selected_method(phase, requested_method, completed_expansions):
    if phase not in ('EARLY', 'GUARD'):
        raise ValueError('Unknown cold search phase')
    if requested_method not in ('beam', 'bc_raw', 'ppo_raw'):
        raise ValueError('Unknown fixed selector')
    if type(completed_expansions) is not int or completed_expansions < 0:
        raise ValueError('Invalid cumulative phase expansion count')
    return requested_method if phase == 'EARLY' and completed_expansions < EARLY_STEPS else 'beam'


def choose_action(search, observation, phase, requested_method, policy=None, generator=None):
    """Return a fixed decision, preserving mask and deterministic tie semantics."""
    method = selected_method(phase, requested_method, search._beam_turn)
    if method == 'beam':
        choice = {'action': search.rule_action('beam'), 'logprob': None, 'value': None}
    else:
        if policy is None:
            raise ValueError('A fixed early policy is required')
        choice = policy.act(observation, observation['mask'], deterministic=True, generator=generator)
        choice = {key: choice[key] for key in ('action', 'logprob', 'value')}
        action = choice['action']
        choice['action'] = action.item() if hasattr(action, 'item') else action
        if not all(math.isfinite(float(choice[key])) for key in ('logprob', 'value')):
            raise CLPUReliabilityError('Nonfinite fixed policy output')
    action = choice['action']
    if type(action) is not int or not 0 <= action < len(observation['mask']) or not observation['mask'][action]:
        raise CLPUReliabilityError('Selector chose an unavailable branch')
    return {**choice, 'method': method, 'policy_used': method != 'beam'}


def complete_plan(result):
    plan = result.get('best_plan')
    if result.get('status') != 'CERTIFIED_INCUMBENT':
        if plan is not None:
            raise CLPUReliabilityError('Uncertified result unexpectedly carries a best plan')
        return None
    if (not isinstance(plan, dict) or plan.get('certified_complete') is not True
            or plan.get('status') != 'COMPLETE' or plan.get('validation', {}).get('status') != 'COMPLETE'):
        raise CLPUReliabilityError('Missing complete plan certificate')
    energy = plan['validation'].get('energy_kwh')
    if not isinstance(energy, (int, float)) or not math.isfinite(energy) or energy < 0:
        raise CLPUReliabilityError('Invalid certified objective')
    return plan


def merge_results(early, guard=None, *, protected):
    """Keep the complete guard on exact energy ties; never excuse a small loss.

The original guard's missing-certificate status is retained for a protected
cell. An independently certified EARLY proposal is still saved in its own
phase result, but does not establish the absent Beam guarantee.
"""
    early_plan = complete_plan(early)
    if not protected:
        if guard is not None:
            raise ValueError('Direct window reference must not hide a guard search')
        chosen, phase, evidence = early, 'EARLY', None
    else:
        if guard is None:
            raise ValueError('Protected output requires a completed independent guard')
        guard_plan = complete_plan(guard)
        established = guard_plan is not None
        chosen, phase = guard, 'GUARD'
        if established and early_plan is not None and early_plan['validation']['energy_kwh'] > guard_plan['validation']['energy_kwh']:
            chosen, phase = early, 'EARLY'
        evidence = {'guard_established': established, 'baseline_status': guard['status'],
                    'baseline_energy_kwh': guard_plan['validation']['energy_kwh'] if established else None,
                    'final_energy_kwh': chosen['best_plan']['validation']['energy_kwh'] if established else None,
                    'nondegradation_verified': established,
                    'tie_rule': 'exact_equal_energy_select_guard',
                    'scope': 'Fixed 30-minute normal-service objective; computation waiting excluded; final completed output only'}
        if established and evidence['final_energy_kwh'] < evidence['baseline_energy_kwh']:
            raise CLPUReliabilityError('Independent Beam floor lost')
    return {'version': VERSION, 'schema': RESULT_SCHEMA, 'status': chosen['status'],
            'selected_phase': phase, 'best_plan': deepcopy(chosen.get('best_plan')),
            'guard_audit': evidence, 'protected': protected,
            'intermediate_beam_floor_claimed': False}
