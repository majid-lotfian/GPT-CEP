import json
import numpy as np
import pandas as pd
import pytest
from gpt_cep.core import Rule, Predicate, evaluate
from gpt_cep.optimization import anneal
from gpt_cep.federated import Client, aggregate, train_federated
from gpt_cep.llm import generate, offline_provider, STRATEGIES
from gpt_cep.metrics import ages_index, srrg

def dataset():
    x = np.arange(100, dtype=float)
    return pd.DataFrame({"heart_rate": x, "activity": (x > 49).astype(int)})

def test_rule_and_optimization():
    df = dataset()
    rule = Rule(1, [Predicate("heart_rate", ">", 30)])
    assert evaluate(rule, df, df.activity)["f1"] < 1
    refined, history = anneal(rule, df[["heart_rate"]], df.activity, steps=1000)
    assert history[-1] >= history[0]
    assert evaluate(refined, df, df.activity)["f1"] >= evaluate(rule, df, df.activity)["f1"]

def test_unsafe_rule_rejected():
    with pytest.raises(ValueError):
        Rule.from_dict({"activity": 1, "predicates": [{"feature": "__import__('os')", "operator": ">", "threshold": 1}]}, ["heart_rate"])

def test_all_prompt_strategies():
    for strategy in STRATEGIES:
        rule = generate(offline_provider, 1, ["heart_rate"], strategy)
        assert rule.activity == 1

def test_federated():
    df = dataset()
    a = Client("a", df.iloc[:60], df.iloc[60:80])
    b = Client("b", df.iloc[20:80], df.iloc[80:])
    rule = Rule(1, [Predicate("heart_rate", ">", 30)])
    merged, reports = train_federated(rule, [a, b], steps=30)
    assert set(reports) == {"a", "b"}
    assert isinstance(merged.predicates[0].threshold, float)

def test_metrics():
    assert srrg(3, 4) == .75
    assert ages_index(.5, .5) == .5
    assert ages_index(0, .5) == 0
