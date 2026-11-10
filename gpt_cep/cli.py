"""Command line experiment runner: python -m gpt_cep.cli ..."""
import argparse
import json
from pathlib import Path
from .core import evaluate, save_rule
from .data import load_pamap2, prepare_split, FEATURES
from .llm import generate, offline_provider, openai_provider
from .optimization import anneal, random_rule

def main():
    parser = argparse.ArgumentParser(description="GPT-CEP rule generation and threshold optimization")
    parser.add_argument("--data", required=True, help="PAMAP2 dataset directory containing subject*.dat")
    parser.add_argument("--activity", type=int, default=4)
    parser.add_argument("--strategy", default="few-shot")
    parser.add_argument("--provider", choices=["offline", "openai", "rfv"], default="offline")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    data = load_pamap2(args.data)
    train, val, test = prepare_split(data)
    if args.provider == "rfv":
        initial = random_rule(args.activity, train[FEATURES])
    else:
        provider = offline_provider if args.provider == "offline" else openai_provider(args.model)
        initial = generate(provider, args.activity, FEATURES, args.strategy)
    refined, history = anneal(initial, train[FEATURES], train.activity, args.steps)
    output = {
        "provider": args.provider, "strategy": args.strategy, "activity": args.activity,
        "initial_validation": evaluate(initial, val[FEATURES], val.activity),
        "refined_validation": evaluate(refined, val[FEATURES], val.activity),
        "initial_test": evaluate(initial, test[FEATURES], test.activity),
        "refined_test": evaluate(refined, test[FEATURES], test.activity),
        "optimization_best_train_f1": history[-1],
        "split_subjects": {"validation": [107], "test": [108, 109]}
    }
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    save_rule(initial, dest / "initial_rule.json")
    save_rule(refined, dest / "refined_rule.json")
    (dest / "metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
