"""Label-free assignment of anonymous observations to a declared CAD inventory.

The global solver is SciPy's exact Jonker--Volgenant linear assignment solver.
It maximizes the ORIGINAL cached score, subject to the declared capacities,
at cardinality min(number of observations, total inventory capacity). It is
not a learned matcher or an open-set detector. Independent selection is the
deliberately unconstrained control. No reference labels enter this module.
"""
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment


def assign(score_matrix, cad_ids, capacities, strategy="global"):
    matrix = np.asarray(score_matrix, dtype=np.float64)
    ids = list(cad_ids)
    counts = np.asarray(capacities)
    if matrix.ndim != 2 or matrix.shape[1] != len(ids):
        raise ValueError("Score matrix and declared inventory disagree")
    if len(ids) != len(set(ids)) or not all(isinstance(x, str) for x in ids):
        raise ValueError("Inventory must contain unique anonymous CAD IDs")
    if counts.shape != (len(ids),) or not np.isfinite(counts).all():
        raise ValueError("Invalid capacities")
    if np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("Capacities must be nonnegative integers")
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite scores: do not silently omit observations")
    order = np.asarray(sorted(range(len(ids)), key=lambda j: ids[j]), dtype=int)
    slot_index = np.repeat(order, counts[order].astype(int))
    predictions = [None] * matrix.shape[0]
    if len(slot_index) == 0 or len(predictions) == 0:
        return {"predictions": predictions, "objective": 0.0,
                "unassigned_inventory": dict(zip(ids, counts.astype(int).tolist()))}
    expanded = matrix[:, slot_index]
    if strategy == "independent":
        rr = np.arange(len(predictions))
        cc = np.argmax(expanded, axis=1)
    elif strategy == "global":
        rr, cc = linear_sum_assignment(expanded, maximize=True)
    elif strategy == "greedy":
        # Stable tie order: incoming observation order, then canonical CAD order.
        used_rows, used_slots, pairs = set(), set(), []
        for flat in np.argsort(-expanded.ravel(), kind="stable"):
            i, j = np.unravel_index(int(flat), expanded.shape)
            if i not in used_rows and j not in used_slots:
                used_rows.add(i)
                used_slots.add(j)
                pairs.append((i, j))
                if len(pairs) == min(expanded.shape):
                    break
        rr = np.asarray([a for a, _ in pairs], dtype=int)
        cc = np.asarray([b for _, b in pairs], dtype=int)
    else:
        raise ValueError("Unknown assignment strategy")
    for i, j in zip(rr, cc):
        predictions[int(i)] = ids[int(slot_index[j])]
    used = Counter(x for x in predictions if x is not None)
    return {"predictions": predictions,
            "objective": float(expanded[rr, cc].sum()),
            "unassigned_inventory": {x: max(int(n) - used[x], 0)
                                     for x, n in zip(ids, counts)}}
