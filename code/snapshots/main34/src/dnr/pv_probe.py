"""D015 fixed-PQ PV input construction; positive demand, positive PV generation."""
import numpy as np


def net_injections(gross_p, gross_q, pv_bus_ids, ratings_mw, profiles):
    p, q = np.asarray(gross_p, dtype=float), np.asarray(gross_q, dtype=float)
    ids = np.asarray(pv_bus_ids, dtype=int)
    ratings, factors = np.asarray(ratings_mw, dtype=float), np.asarray(profiles, dtype=float)
    if p.shape != (33,) or q.shape != (33,) or not np.isfinite([p, q]).all():
        raise ValueError('Expected finite 33-bus gross load vectors')
    if (p < 0).any() or (q < 0).any() or p[0] != 0 or q[0] != 0:
        raise ValueError('Invalid gross load or nonzero source load')
    if ids.ndim != 1 or not (len(ids) == len(ratings) == len(factors)):
        raise ValueError('PV mapping dimension mismatch')
    if len(set(ids.tolist())) != len(ids) or (ids < 2).any() or (ids > 33).any():
        raise ValueError('PV bus ids must be unique original non-source bus ids')
    if not np.isfinite([ratings, factors]).all() or (ratings < 0).any() or (factors < 0).any() or (factors > 1).any():
        raise ValueError('PV rating/profile outside declared domain')
    pv = np.zeros(33)
    pv[ids - 1] = ratings * factors
    return p - pv, q.copy(), pv
