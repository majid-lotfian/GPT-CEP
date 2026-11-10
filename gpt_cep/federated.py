"""Privacy-preserving rule-parameter aggregation prototype.
Only rule parameters and aggregate metrics leave clients; not raw event records.
This is a research implementation, not secure aggregation or differential privacy.
"""
from dataclasses import dataclass
import numpy as np
from .core import Rule, evaluate
from .optimization import anneal

@dataclass
class Client:
    client_id: str
    train: object
    validation: object

    def refine(self, rule, steps=200, seed=42):
        return anneal(rule, self.train, self.train["activity"], steps, seed)[0]

def aggregate(rules, weights=None):
    if not rules:
        raise ValueError("No client rules")
    first = rules[0]
    signature = [(p.feature, p.operator) for p in first.predicates]
    if any(r.activity != first.activity or r.conjunction != first.conjunction or [(p.feature, p.operator) for p in r.predicates] != signature for r in rules):
        raise ValueError("Only structurally compatible rules can be aggregated")
    weights = np.asarray(weights if weights is not None else [1] * len(rules), dtype=float)
    if len(weights) != len(rules) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Invalid aggregation weights")
    result = Rule.from_dict(first.to_dict(), [p.feature for p in first.predicates])
    for i, pred in enumerate(result.predicates):
        pred.threshold = float(np.average([r.predicates[i].threshold for r in rules], weights=weights))
    return result

def train_federated(initial, clients, steps=200, seed=42):
    """Local refinement -> compatible rule aggregation -> independent client validation."""
    local = [client.refine(initial, steps, seed + i) for i, client in enumerate(clients)]
    global_rule = aggregate(local, [len(c.train) for c in clients])
    reports = {c.client_id: evaluate(global_rule, c.validation, c.validation["activity"]) for c in clients}
    return global_rule, reports
