"""FedAvg aggregation, ML doc §8. Pure list-based averaging (no torch
needed here -- state_dicts arrive as plain JSON nested lists) so it's
simple to unit test independent of the model architecture.
"""


def tree_mean(values: list):
    """Elementwise mean across N same-shaped nested lists (a state_dict
    tensor exported as nested lists), recursing until scalars.
    """
    if isinstance(values[0], list):
        return [tree_mean([v[i] for v in values]) for i in range(len(values[0]))]
    return sum(values) / len(values)


def fedavg(state_dicts: list[dict]) -> dict:
    """Each agent 'cluster' trains locally and sends only weights (system
    doc §4.6) -- this is the aggregation server's merge step. Simple
    unweighted average across clients (this system has no per-client
    sample-count metadata to weight by, a documented simplification).
    """
    if not state_dicts:
        raise ValueError("fedavg requires at least one client state_dict")
    keys = state_dicts[0].keys()
    for sd in state_dicts:
        if sd.keys() != keys:
            raise ValueError("all client state_dicts must share the same architecture/keys")
    return {key: tree_mean([sd[key] for sd in state_dicts]) for key in keys}
