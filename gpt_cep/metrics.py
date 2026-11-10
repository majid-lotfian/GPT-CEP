"""Paper-inspired rule-generation reliability and AGES index."""
import numpy as np

def srrg(correct_responses, total_prompts):
    if total_prompts <= 0 or not 0 <= correct_responses <= total_prompts:
        raise ValueError("Invalid response counts")
    return correct_responses / total_prompts

def generation_efficiency(normalized_srrg, normalized_time, normalized_cost):
    if normalized_time <= 0 or normalized_cost <= 0:
        raise ValueError("Denominators must be positive")
    return normalized_srrg / (normalized_time * normalized_cost)

def ages_index(normalized_accuracy, normalized_ge, normalized_model_size=0., alpha=0.):
    if not all(0 <= v <= 1 for v in (normalized_accuracy, normalized_ge, normalized_model_size, alpha)):
        raise ValueError("AGES inputs must be normalized to [0, 1]")
    denom = normalized_accuracy + normalized_ge
    harmonic = 2 * normalized_accuracy * normalized_ge / denom if denom else 0.
    return harmonic * (1 - alpha * normalized_model_size)

def minmax(values):
    x = np.asarray(values, dtype=float)
    spread = x.max() - x.min()
    return np.zeros_like(x) if spread == 0 else (x - x.min()) / spread
