# GPT-CEP — Adaptive LLM-Assisted Complex Event Processing

**GPT-CEP** investigates how large language models can generate interpretable event-detection rules from user intent, and how locally observed sensor data can improve rule thresholds through simulated annealing.

This repository contains **the original DEBS research scripts** under `code/` and a **new, installable research pipeline** under `gpt_cep/`. The new pipeline makes the components easier to run, test and extend. It is **not a claim that all paper experiments have been reproduced** or that the system is a production CEP deployment.

## Architecture

```text
Natural-language activity specification
                 |
      LLM + prompting strategy
      (or deterministic offline mock)
                 |
        JSON rule validation
                 |
      Initial interpretable rule
                 |
    +------------+-------------+
    |                          |
 Local client A             Local client B
 Simulated annealing        Simulated annealing
 (training data only)       (training data only)
    |                          |
    +------ parameter-only ----+
              aggregation
                 |
       Validation per client
                 |
       Candidate refined rule
                 |
        Rule export (JSON)
```

The new code includes the following research components:

| Component | Module | Status |
| --- | --- | --- |
| JSON rule representation, allowlisted operators, validation | `core.py` | Implemented |
| PAMAP2 preprocessing and subject-level train/validation/test split | `data.py` | Implemented |
| Seven prompting strategies | `llm.py` | Implemented prompt templates |
| OpenAI adapter and offline smoke-test provider | `llm.py` | Implemented |
| Simulated annealing local threshold tuning | `optimization.py` | Implemented |
| RFV random-feature/random-threshold baseline | `optimization.py` | Implemented |
| Client-local optimization and weighted parameter aggregation | `federated.py` | Implemented research prototype |
| SRRG, generation efficiency and AGES scoring functions | `metrics.py` | Implemented |
| CLI, JSON metrics, automated unit tests | `cli.py`, `tests/` | Implemented |
| RLP full model/prompt random-selection experiment | Original scripts/paper | Not yet integrated into new CLI |
| Real Flink/Kafka CEP execution, secure aggregation, differential privacy | — | Not implemented |
| Full paper figure/table reproduction and published F1 verification | — | Requires original datasets, prompts, model versions and evaluation settings |

### Important research distinction

The **DEBS 2024 poster** presents an initial LLM-powered rule-generation and federated refinement design with preliminary PAMAP2 experiments. The **DEBS 2025 submitted manuscript** extends the architecture with cluster-based optimization, the AGES Index, and simulated annealing. In the 2025 evaluation, the rule-aggregation module was **explicitly bypassed** for the single-client threshold-refinement comparison. The parameter-aggregation implementation in this repository is a subsequent prototype, **not a reproduced result from that experiment**.

## Installation

Python 3.10+:

```bash
git clone https://github.com/majid-lotfian/GPT-CEP.git
cd GPT-CEP
python -m pip install -e ".[test]"
pytest -q
```

For API-backed rule generation:

```bash
python -m pip install -e ".[llm]"
export OPENAI_API_KEY="your-key"
```

Do not commit API keys or private event data.

## Dataset

Download **PAMAP2 Physical Activity Monitoring** from the [UCI repository](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring), and extract it locally. Point `--data` to a directory containing `subject101.dat` ... `subject109.dat`, typically `PAMAP2_Dataset/Protocol`. The loader uses heart rate and three accelerometer magnitudes and keeps six activities: lying (1), sitting (2), standing (3), walking (4), running (5), cycling (6).

**Split:** subjects 101–106 for training, 107 for validation, 108–109 for test (when available). Missing sensor values are imputed using training-set medians only. This differs from some original scripts, which used a random row split and evaluated simulated annealing on the test set. The new implementation avoids that leakage; consequently its metrics are **not directly comparable** to the paper without matching experimental conditions.

## Quick start

Offline deterministic smoke test (no LLM API required):

```bash
gpt-cep --data /path/to/PAMAP2_Dataset/Protocol --activity 4 --provider offline --steps 300 --out results/walking
```

Real LLM generation (requires credentials and API access):

```bash
gpt-cep --data /path/to/PAMAP2_Dataset/Protocol --activity 4 --provider openai --model gpt-4o-mini --strategy few-shot --out results/walking-openai
```

RFV baseline:

```bash
gpt-cep --data /path/to/PAMAP2_Dataset/Protocol --activity 4 --provider rfv --out results/walking-rfv
```

The command writes `initial_rule.json`, `refined_rule.json` and `metrics.json`. Offline mode uses a **fixed JSON stub** and must never be presented as an LLM experiment.

## Federated prototype

```python
from gpt_cep.core import Rule, Predicate
from gpt_cep.federated import Client, train_federated

initial = Rule(activity=4, predicates=[Predicate("heart_rate", ">", 100)])
clients = [
    Client("site-a", site_a_train, site_a_validation),
    Client("site-b", site_b_train, site_b_validation),
]
refined, per_site_metrics = train_federated(initial, clients, steps=300)
```

Each client keeps its rows local and returns optimized thresholds. The aggregator computes a weighted average of **structurally compatible** rules; no raw records are exchanged through the aggregator API. This is **not cryptographically secure federated learning** and does not protect against information leakage from shared model parameters.

## Prompting and metrics

Supported prompt modes: `zero-shot`, `zero-shot-cot`, `zero-shot-tot`, `few-shot`, `few-shot-cot`, `few-shot-cot-sc`, `few-shot-tot`. The `cot` and `tot` variants are prompt instructions, **not independently verified multi-branch search algorithms**. For faithful self-consistency experiments, collect multiple responses and vote externally.

The manuscript defines **SRRG** as valid responses divided by prompts, **generation efficiency** from normalized SRRG, time and monetary impact, and **AGES** as the harmonic mean of normalized accuracy and generation efficiency with an optional model-size adjustment. The functions in `metrics.py` expose these calculations. **Input normalization, evaluation protocols and model-cost estimates must be recorded** for meaningful model comparisons.

## Original scripts

The `code/rule refinement/` directory contains the historical per-activity RFV, RLP and GPT-CEP simulated-annealing scripts. They are retained for provenance, including their original Windows dataset paths and experimental setup. They are **not the supported CLI**; use the package above for portable execution.

## Verification and limitations

Run `pytest -q` to check rule evaluation, safe parsing, prompting, optimization, aggregation and metric functions. These are unit/synthetic tests; passing them does not establish reproduction of published figures or a production-ready streaming system. The current implementation uses tabular, batch event detection, not a deployed Kafka/Flink streaming runtime.

## Research references

- Lotfian Delouee, M., Pernes, D. G., Degeler, V., and Koldehofe, B. (2024). *Towards Federated LLM-Powered CEP Rule Generation and Refinement*. DEBS '24. DOI: [10.1145/3629104.3672429](https://doi.org/10.1145/3629104.3672429).
- *GPT-CEP: Adaptive and Autonomous Rule Learning with LLMs in Federated Complex Event Processing Systems*. DEBS 2025 submitted manuscript (supplied by the authors; submission status is not publication confirmation).

## License

See [LICENSE](LICENSE).
