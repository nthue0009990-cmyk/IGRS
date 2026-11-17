"""Preparation-only reservations for NC123; no engine import or process launch.

This counter does not instrument DSS, supervise memory, or authorize execution.
A future adapter must reserve before each numerical entry, including initialization.
"""
from __future__ import annotations

import math
import time


class AcceptanceBudgetStop(RuntimeError):
    pass


class AcceptanceBudget:
    def __init__(self, *, max_requests=16, max_entries=256, seconds=240,
                 clock=time.monotonic, emit=None):
        for name, value in [('max_requests', max_requests), ('max_entries', max_entries)]:
            if type(value) is not int or value < 1:
                raise ValueError(name + ' must be a positive integer')
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('A finite positive wall limit is required')
        self.clock, self.emit = clock, emit or (lambda row: None)
        self.started = self._time()
        self.deadline = self.started + seconds
        self.last_time = self.started
        self.max_requests, self.max_entries = max_requests, max_entries
        self.requests, self.entries = {}, {}
        self.reserved_entries = self.returned_entries = 0
        self.next_id = 0

    def _time(self):
        value = self.clock()
        if type(value) not in (int, float) or not math.isfinite(value):
            raise AcceptanceBudgetStop('Invalid monotonic clock')
        return value

    def check(self):
        now = self._time()
        if now < self.last_time:
            raise AcceptanceBudgetStop('Clock moved backwards')
        self.last_time = now
        if now >= self.deadline:
            raise AcceptanceBudgetStop('Wall budget reached')

    def begin_request(self, request_id, request_sha256):
        self.check()
        if not isinstance(request_id, str) or not request_id:
            raise ValueError('Nonempty request identity required')
        if (not isinstance(request_sha256, str) or len(request_sha256) != 64
                or any(c not in '0123456789abcdef' for c in request_sha256)):
            raise ValueError('Canonical request SHA-256 required')
        if request_id in self.requests:
            raise AcceptanceBudgetStop('Duplicate request identity; no implicit retry')
        if len(self.requests) >= self.max_requests:
            raise AcceptanceBudgetStop('Request budget reached')
        self.requests[request_id] = request_sha256
        # Charges persist even if durable logging fails; caller must stop.
        self.emit({'event': 'request_reserved', 'request_id': request_id,
                   'request_sha256': request_sha256, 'request_number': len(self.requests)})

    def reserve_entry(self, *, operation, count, request_id=None):
        self.check()
        if not isinstance(operation, str) or not operation:
            raise ValueError('Named numerical operation required')
        if type(count) is not int or count < 1:
            raise ValueError('Positive explicit numerical entry count required')
        if request_id is not None and request_id not in self.requests:
            raise ValueError('Numerical operation must bind a known request or initialization')
        if self.reserved_entries + count > self.max_entries:
            raise AcceptanceBudgetStop('Numerical entry budget would be exceeded')
        token = self.next_id
        self.next_id += 1
        row = {'event': 'numerical_entries_reserved', 'token': token,
               'operation': operation, 'count': count, 'request_id': request_id,
               'phase': 'initialization' if request_id is None else 'snapshot',
               'completion': 'NOT_REPORTED'}
        self.reserved_entries += count
        self.entries[token] = dict(row)
        self.emit(dict(row))
        return token

    def finish_entry(self, token, *, returned):
        # Even after timeout, preserve the observed return/failure without refund.
        if type(token) is not int or token not in self.entries or type(returned) is not bool:
            raise ValueError('Known token and explicit returned flag required')
        row = self.entries[token]
        if row['completion'] != 'NOT_REPORTED':
            raise ValueError('Numerical completion already recorded')
        row['completion'] = 'API_RETURNED_NOT_CONVERGENCE' if returned else 'UNKNOWN_NO_REFUND'
        if returned:
            self.returned_entries += row['count']
        self.emit(dict(row))

    def snapshot(self):
        return {'requests': len(self.requests), 'reserved_entries': self.reserved_entries,
                'returned_entries': self.returned_entries,
                'unreported_entries': sum(r['count'] for r in self.entries.values()
                                          if r['completion'] == 'NOT_REPORTED'),
                'max_requests': self.max_requests, 'max_entries': self.max_entries,
                'deadline': self.deadline, 'engine_instrumentation_implemented': False,
                'memory_supervisor_implemented': False,
                'physical_certificate_created': False}
