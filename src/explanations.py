"""Explanation Generation and Confidence Scoring Engine (Bonus Track).

Generates natural-language justifications for eviction decisions and assesses
the calibration between self-rated confidence scores and empirical correctness.
"""

from typing import Dict, List, Any, Tuple
import numpy as np


class EvictionExplanationEngine:
    """Produces natural-language explanations and calibrated confidence scores for page evictions."""

    def __init__(self, high_confidence_margin: float = 0.25):
        self.high_confidence_margin = high_confidence_margin

    def generate_explanation(
        self,
        evicted_page: int,
        candidates: List[int],
        scores: np.ndarray,
        features_dict: Dict[int, np.ndarray],
        optimal_page: int,
        timestep: int,
    ) -> Dict[str, Any]:
        """Generate explanation, confidence score, and evaluate correctness.

        Args:
            evicted_page: The page selected for eviction.
            candidates: List of all resident pages in cache.
            scores: Model eviction probability scores for each candidate.
            features_dict: Dict mapping page_id -> feature array [recency, freq, age, prev_dist].
            optimal_page: The ground-truth Belady optimal page to evict.
            timestep: Current memory reference index.

        Returns:
            Dict containing explanation string, confidence score, and correctness boolean.
        """
        sorted_indices = np.argsort(scores)[::-1]
        top_idx = sorted_indices[0]
        runner_up_idx = sorted_indices[1] if len(sorted_indices) > 1 else top_idx

        top_score = float(scores[top_idx])
        runner_up_score = float(scores[runner_up_idx])
        margin = top_score - runner_up_score

        # Self-rated confidence: maps margin into [0.50, 0.99]
        # Base probability scaled by margin sharpness
        confidence = float(np.clip(0.50 + 0.49 * np.tanh(margin * 3.5), 0.50, 0.99))

        # Check correctness against Belady's Optimal choice
        is_correct = (evicted_page == optimal_page)

        # Feature interpretation
        ev_feat = features_dict[evicted_page]
        recency, freq, age, prev_dist = ev_feat[0], ev_feat[1], ev_feat[2], ev_feat[3]

        runner_up_page = candidates[runner_up_idx]
        ru_feat = features_dict[runner_up_page]

        # Natural language synthesis
        if margin > self.high_confidence_margin:
            conf_level = "High confidence"
            comparison = (
                f"Page {evicted_page} exhibits strong eviction indicators (idle for {int(recency)} ticks, "
                f"lifetime frequency of {int(freq)}, cache age {int(age)}), significantly exceeding "
                f"contender Page {runner_up_page} (idle {int(ru_feat[0])} ticks, frequency {int(ru_feat[1])})."
            )
        elif margin > 0.08:
            conf_level = "Moderate confidence"
            comparison = (
                f"Page {evicted_page} selected over Page {runner_up_page} due to higher inactivity "
                f"({int(recency)} vs {int(ru_feat[0])} ticks), though access frequencies are comparable "
                f"({int(freq)} vs {int(ru_feat[1])})."
            )
        else:
            conf_level = "Low confidence (close margin)"
            comparison = (
                f"Marginal distinction between Page {evicted_page} and Page {runner_up_page}. "
                f"Model identifies similar aging profiles ({int(age)} vs {int(ru_feat[2])} ticks); "
                f"eviction relies primarily on recent reuse interval ({int(prev_dist)} vs {int(ru_feat[3])})."
            )

        explanation_text = (
            f"[{conf_level} ({confidence:.2f})] Evicted Page {evicted_page} at tick {timestep}. "
            f"{comparison} Ground-truth Belady agreement: {'YES' if is_correct else 'NO'}."
        )

        return {
            "timestep": timestep,
            "evicted_page": evicted_page,
            "optimal_page": optimal_page,
            "confidence": confidence,
            "is_correct": is_correct,
            "margin": margin,
            "explanation": explanation_text,
        }

    @staticmethod
    def compute_calibration_metrics(
        confidences: List[float],
        correctness: List[bool],
        num_bins: int = 10,
    ) -> Dict[str, Any]:
        """Compute Expected Calibration Error (ECE) and reliability diagram data."""
        conf_arr = np.array(confidences)
        corr_arr = np.array(correctness, dtype=float)

        bin_boundaries = np.linspace(0.5, 1.0, num_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        ece = 0.0
        total_samples = len(conf_arr)

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (conf_arr >= bin_lower) & (conf_arr < bin_upper if i < num_bins - 1 else conf_arr <= bin_upper)
            count = int(np.sum(in_bin))
            bin_counts.append(count)

            if count > 0:
                acc = float(np.mean(corr_arr[in_bin]))
                conf = float(np.mean(conf_arr[in_bin]))
                bin_accuracies.append(acc)
                bin_confidences.append(conf)
                ece += (count / total_samples) * abs(acc - conf)
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append((bin_lower + bin_upper) / 2.0)

        # Brier Score: Mean squared error of probabilistic predictions
        brier_score = float(np.mean((conf_arr - corr_arr) ** 2)) if total_samples > 0 else 0.0

        return {
            "ece": float(ece),
            "brier_score": brier_score,
            "bin_boundaries": bin_boundaries.tolist(),
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
            "total_evaluations": total_samples,
            "overall_accuracy": float(np.mean(corr_arr)) if total_samples > 0 else 0.0,
            "mean_confidence": float(np.mean(conf_arr)) if total_samples > 0 else 0.0,
        }
