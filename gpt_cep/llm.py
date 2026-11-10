"""LLM prompting and safe JSON rule extraction.
The provider is injectable; the offline mock supports CI without API credentials.
"""
import json
import re
from .core import Rule

STRATEGIES = ("zero-shot", "zero-shot-cot", "zero-shot-tot", "few-shot", "few-shot-cot", "few-shot-cot-sc", "few-shot-tot")

def build_prompt(activity, features, strategy="few-shot", examples=None):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    instruction = ("Return ONLY a JSON object with keys activity (integer), conjunction ('all' or 'any'), "
                   "and predicates (array of objects with feature, operator and numeric threshold). "
                   "Use only provided features and operators <, <=, >, >=. No executable code. ")
    context = f"Target activity ID: {activity}. Available sensor features: {', '.join(features)}. "
    if strategy.startswith("few-shot"):
        context += "Examples: " + json.dumps(examples or [{"activity": activity, "conjunction": "all", "predicates": [{"feature": features[0], "operator": ">", "threshold": 0.0}]}]) + ". "
    if "cot" in strategy:
        context += "Check the constraints internally before giving only the final JSON. "
    if "tot" in strategy:
        context += "Compare alternative feature/threshold combinations internally; return the best final JSON. "
    return instruction + context

def parse_response(response, features):
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^\x60{3}(?:json)?\s*|\s*\x60{3}$", "", text).strip()
    return Rule.from_dict(json.loads(text), features)

def generate(provider, activity, features, strategy="few-shot", examples=None):
    return parse_response(provider(build_prompt(activity, features, strategy, examples)), features)

def openai_provider(model="gpt-4o-mini"):
    """Optional OpenAI API adapter. Set OPENAI_API_KEY in the environment."""
    from openai import OpenAI
    client = OpenAI()
    def complete(prompt):
        result = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0)
        return result.choices[0].message.content
    return complete

def offline_provider(prompt):
    """Deterministic smoke-test generator, not an LLM and not a research result."""
    match = re.search(r"Target activity ID: (\d+)", prompt)
    activity = int(match.group(1))
    features = re.search(r"Available sensor features: ([^.]+)\.", prompt).group(1).split(", ")
    return json.dumps({"activity": activity, "conjunction": "all", "predicates": [{"feature": features[0], "operator": ">", "threshold": 80.0}]})
