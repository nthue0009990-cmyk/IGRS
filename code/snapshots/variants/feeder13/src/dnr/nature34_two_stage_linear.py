"""Sparse affine formulation utilities; no electrical model is implied here."""
from dataclasses import dataclass
import math
import time
import warnings


@dataclass(frozen=True)
class Expr:
    terms: dict
    constant: float = 0.

    def __add__(self, other):
        other = expression(other)
        terms = dict(self.terms)
        for key, value in other.terms.items():
            terms[key] = terms.get(key, 0.) + value
            if terms[key] == 0:
                del terms[key]
        return Expr(terms, self.constant + other.constant)

    __radd__ = __add__

    def __neg__(self):
        return self * -1.

    def __sub__(self, other):
        return self + -expression(other)

    def __rsub__(self, other):
        return expression(other) + -self

    def __mul__(self, value):
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            raise TypeError('Only finite scalar multiplication is affine')
        return Expr({k: v * value for k, v in self.terms.items() if v * value},
                    self.constant * value)

    __rmul__ = __mul__

    def __truediv__(self, value):
        return self * (1. / value)

    def value(self, x):
        return math.fsum([self.constant] + [v * float(x[k]) for k, v in self.terms.items()])


def expression(value):
    if isinstance(value, Expr):
        return value
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TypeError('A finite scalar or affine expression is required')
    return Expr({}, float(value))


class LinModel:
    def __init__(self):
        self.names, self.lower, self.upper, self.integer = [], [], [], []
        self.rows = []
        self._names = set()

    def var(self, name, lb, ub, integer=False):
        if name in self._names or not all(math.isfinite(v) for v in (lb, ub)) or lb > ub:
            raise ValueError('Unique name and finite ordered variable bounds required')
        index = len(self.names)
        self._names.add(name)
        self.names.append(name)
        self.lower.append(float(lb))
        self.upper.append(float(ub))
        self.integer.append(int(bool(integer)))
        return Expr({index: 1.})

    def bounds(self, expr):
        expr = expression(expr)
        lo, hi = expr.constant, expr.constant
        for k, v in expr.terms.items():
            a, b = v * self.lower[k], v * self.upper[k]
            lo += min(a, b)
            hi += max(a, b)
        return lo, hi

    def add(self, expr, lb=None, ub=None, name=''):
        expr = expression(expr)
        lo = -math.inf if lb is None else float(lb)
        hi = math.inf if ub is None else float(ub)
        if math.isnan(lo) or math.isnan(hi) or lo > hi:
            raise ValueError('Invalid row bounds')
        self.rows.append((expr, lo, hi, str(name)))

    def _binary(self, value):
        value = expression(value)
        if not value.terms and value.constant in (0., 1.):
            return value
        if len(value.terms) != 1 or value.constant != 0:
            raise ValueError('A binary variable or constant is required')
        k, c = next(iter(value.terms.items()))
        if c != 1. or self.integer[k] != 1 or self.lower[k] != 0 or self.upper[k] != 1:
            raise ValueError('A binary variable or constant is required')
        return value

    def product(self, binary, expr, name):
        binary, expr = self._binary(binary), expression(expr)
        if not binary.terms:
            return binary.constant * expr
        if not expr.terms:
            return expr.constant * binary
        lo, hi = self.bounds(expr)
        result = self.var(name, min(0., lo), max(0., hi))
        self.add(result - lo * binary, lb=0, name=name + ':lower_off')
        self.add(result - hi * binary, ub=0, name=name + ':upper_off')
        self.add(result - expr - hi * binary, lb=-hi, name=name + ':lower_on')
        self.add(result - expr - lo * binary, ub=-lo, name=name + ':upper_on')
        return result

    def implies(self, binary, expr, lb=None, ub=None, name=''):
        binary, expr = self._binary(binary), expression(expr)
        if not binary.terms:
            if binary.constant == 1:
                self.add(expr, lb, ub, name)
            return
        lo, hi = self.bounds(expr)
        if lb is not None:
            margin = max(0., float(lb) - lo)
            self.add(expr - margin * binary, lb=float(lb) - margin, name=name + ':lo')
        if ub is not None:
            margin = max(0., hi - float(ub))
            self.add(expr + margin * binary, ub=float(ub) + margin, name=name + ':hi')

    def residuals(self, values):
        if len(values) != len(self.names) or any(not math.isfinite(v) for v in values):
            raise ValueError('Finite complete solution required')
        variable = max([0.] + [max(lo - x, x - hi, 0.)
                       for x, lo, hi in zip(values, self.lower, self.upper)])
        integrality = max([0.] + [abs(x - round(x)) for x, flag in zip(values, self.integer) if flag])
        worst, row_name = 0., None
        for expr, lo, hi, name in self.rows:
            v = expr.value(values)
            error = max(lo - v, v - hi, 0.)
            if error > worst:
                worst, row_name = error, name
        return {'max_variable_bound_violation': variable, 'max_integer_violation': integrality,
                'max_row_violation': worst, 'worst_row': row_name}

    def solve(self, objective, *, time_limit, node_limit, mip_rel_gap=0.):
        """One explicitly budgeted optimization call; caller records all costs."""
        started, cpu = time.perf_counter(), time.process_time()
        import numpy as np
        from scipy.optimize import milp, Bounds, LinearConstraint
        from scipy.sparse import coo_matrix
        if time_limit <= 0 or node_limit <= 0 or mip_rel_gap < 0:
            raise ValueError('Explicit positive solver budget required')
        objective = expression(objective)
        c = np.zeros(len(self.names))
        for k, v in objective.terms.items():
            c[k] = v
        ii, jj, data, low, high = [], [], [], [], []
        for row, (expr, lo, hi, _) in enumerate(self.rows):
            low.append(lo - expr.constant)
            high.append(hi - expr.constant)
            for k, v in expr.terms.items():
                ii.append(row); jj.append(k); data.append(v)
        matrix = coo_matrix((data, (ii, jj)), shape=(len(self.rows), len(self.names))).tocsc()
        remaining = time_limit - (time.perf_counter() - started)
        if remaining <= 0:
            return {'status':1, 'message':'TIME_LIMIT_DURING_SPARSE_MATRIX_BUILD', 'x':None,
                    'objective':None, 'objective_constant':objective.constant,
                    'mip_dual_bound':None, 'backend_raw_dual_bound':None, 'mip_gap':None,
                    'gap_scope':'backend not called', 'mip_node_count':None,
                    'wall_seconds':time.perf_counter()-started, 'cpu_seconds':time.process_time()-cpu,
                    'variable_count':len(self.names),'row_count':len(self.rows),'options':None,
                    'warnings':[],'residuals':None,'physical_certificate':None}
        options = {'time_limit':float(remaining), 'node_limit':int(node_limit),
                   'mip_rel_gap':float(mip_rel_gap), 'threads':1}
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            result = milp(c, integrality=np.asarray(self.integer),
                          bounds=Bounds(self.lower, self.upper),
                          constraints=LinearConstraint(matrix, low, high), options=options)
        x = result.x.tolist() if result.x is not None else None
        def finite(name):
            value = getattr(result, name, None)
            return float(value) if value is not None and math.isfinite(value) else None
        raw_bound = finite('mip_dual_bound')
        return {'status':int(result.status), 'message':str(result.message), 'x':x,
                'objective':None if x is None else objective.value(x),
                'objective_constant':objective.constant,
                'mip_dual_bound':None if raw_bound is None else raw_bound + objective.constant,
                'backend_raw_dual_bound':raw_bound, 'mip_gap':finite('mip_gap'),
                'gap_scope':'backend objective excluding the reported objective_constant',
                'mip_node_count':finite('mip_node_count'),
                'wall_seconds':time.perf_counter()-started, 'cpu_seconds':time.process_time()-cpu,
                'variable_count':len(self.names), 'row_count':len(self.rows), 'options':options,
                'warnings':[str(w.message) for w in seen],
                'residuals':None if x is None else self.residuals(x),
                'physical_certificate':None}

    def to_record(self):
        return {'variables':[{'name':n,'lb':lo,'ub':hi,'integer':i}
                             for n,lo,hi,i in zip(self.names,self.lower,self.upper,self.integer)],
                'rows':[{'name':n,'terms':sorted(e.terms.items()),'constant':e.constant,
                         'lb':lo if math.isfinite(lo) else None,
                         'ub':hi if math.isfinite(hi) else None} for e,lo,hi,n in self.rows]}
