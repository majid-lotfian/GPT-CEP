"""Local threshold optimization using simulated annealing, evaluated on training only."""
import math
import numpy as np
from .core import Rule, Predicate, evaluate

def anneal(rule, frame, labels, steps=300, seed=42, temperature=1.0):
    if steps < 1:
        raise ValueError("steps must be positive")
    rng = np.random.default_rng(seed)
    current = Rule.from_dict(rule.to_dict(), list(frame.columns))
    best = Rule.from_dict(rule.to_dict(), list(frame.columns))
    score = evaluate(current, frame, labels)["f1"]
    best_score = score
    scales = {p.feature: max(float(frame[p.feature].std()), 1e-3) for p in rule.predicates}
    history = [score]
    for i in range(steps):
        proposal = Rule.from_dict(current.to_dict(), list(frame.columns))
        j = int(rng.integers(len(proposal.predicates)))
        p = proposal.predicates[j]
        p.threshold += float(rng.normal(0, scales[p.feature] * 0.2))
        new_score = evaluate(proposal, frame, labels)["f1"]
        temp = max(temperature * (0.01 ** (i / steps)), 1e-6)
        if new_score >= score or rng.random() < math.exp((new_score - score) / temp):
            current, score = proposal, new_score
        if score > best_score:
            best, best_score = Rule.from_dict(current.to_dict(), list(frame.columns)), score
        history.append(best_score)
    return best, history

def random_rule(activity, frame, seed=42, count=3):
    rng = np.random.default_rng(seed)
    features = list(frame.columns)
    chosen = rng.choice(features, size=min(count, len(features)), replace=False)
    return Rule(int(activity), [Predicate(str(f), str(rng.choice(["<", ">"])), float(frame[f].quantile(rng.uniform(.2, .8)))) for f in chosen])
