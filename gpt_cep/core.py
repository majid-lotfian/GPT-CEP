"""Validated, data-driven CEP rules; no execution of model-generated Python."""
from dataclasses import dataclass, asdict
from typing import Any
import json
import numpy as np

OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}

@dataclass
class Predicate:
    feature: str
    operator: str
    threshold: float

@dataclass
class Rule:
    activity: int
    predicates: list[Predicate]
    conjunction: str = "all"
    name: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], features: list[str]):
        if data.get("conjunction", "all") not in ("all", "any"):
            raise ValueError("conjunction must be all or any")
        preds = []
        for p in data["predicates"]:
            if p["feature"] not in features or p["operator"] not in OPS:
                raise ValueError("unknown feature/operator")
            t = float(p["threshold"])
            if not np.isfinite(t):
                raise ValueError("non-finite threshold")
            preds.append(Predicate(p["feature"], p["operator"], t))
        if not preds or len(preds) > 24:
            raise ValueError("expected 1..24 predicates")
        return cls(int(data["activity"]), preds, data.get("conjunction", "all"), str(data.get("name", "")))

    def to_dict(self):
        return asdict(self)

    def predict(self, frame):
        masks = []
        for p in self.predicates:
            masks.append(OPS[p.operator](frame[p.feature].to_numpy(dtype=float), p.threshold))
        return (np.logical_and.reduce(masks) if self.conjunction == "all" else np.logical_or.reduce(masks)).astype(int)

    def classify(self, frame):
        return np.where(self.predict(frame).astype(bool), self.activity, 0)

def binary_metrics(truth, predicted):
    truth, predicted = np.asarray(truth).astype(bool), np.asarray(predicted).astype(bool)
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

def evaluate(rule, frame, labels):
    return binary_metrics(np.asarray(labels) == rule.activity, rule.predict(frame))

def save_rule(rule, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rule.to_dict(), f, indent=2)
