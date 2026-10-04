"""Simulation Engine and Metric Collection for Page Replacement Experiments.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

from src.algorithms import (
    FIFOPolicy,
    LRUPolicy,
    OptimalPolicy,
    LearnedPolicy,
    AdaptiveHybridPolicy,
)
from src.models import LogisticRegressionClassifier, DecisionTreeClassifierCustom
from src.explanations import EvictionExplanationEngine


class Simulator:
    """Manages training data collection, model training, and comparative policy simulations."""

    def __init__(self, capacity: int, seed: int = 42):
        self.capacity = capacity
        self.seed = seed
        self.explanation_engine = EvictionExplanationEngine()

    def generate_training_data_from_oracle(
        self,
        train_trace: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Runs Belady's algorithm on a training trace to extract (features, label) pairs.

        For each eviction event:
        - Resident page with longest forward reuse distance -> label 1 (ideal eviction)
        - Resident pages with immediate near reuse -> label 0 (retain in cache)
        """
        cache: Dict[int, Dict[str, Any]] = {}
        X_list = []
        y_list = []

        for t, page in enumerate(train_trace):
            if page in cache:
                cache[page]["prev_dist"] = t - cache[page]["last_access"]
                cache[page]["last_access"] = t
                cache[page]["access_count"] += 1
                continue

            if len(cache) >= self.capacity:
                future_subtrace = train_trace[t + 1 :]
                next_positions = {}
                for idx, p in enumerate(future_subtrace):
                    if p in cache and p not in next_positions:
                        next_positions[p] = idx

                # Determine Belady candidate
                furthest_dist = -1
                optimal_evicted = None
                for resident in cache:
                    dist = next_positions.get(resident, 999999)
                    if dist > furthest_dist:
                        furthest_dist = dist
                        optimal_evicted = resident

                # Record feature vectors for all resident pages
                for resident, meta in cache.items():
                    recency = float(t - meta["last_access"])
                    frequency = float(meta["access_count"])
                    age = float(t - meta["load_time"])
                    prev_dist = float(meta.get("prev_dist", recency))
                    feat = [recency, frequency, age, prev_dist]

                    # Label: 1 if this was optimal eviction or reuse distance > 100, else 0
                    is_optimal = 1 if (resident == optimal_evicted or next_positions.get(resident, 999999) > 120) else 0

                    X_list.append(feat)
                    y_list.append(is_optimal)

                del cache[optimal_evicted]

            cache[page] = {
                "load_time": t,
                "last_access": t,
                "access_count": 1,
                "prev_dist": 1.0,
            }

        return np.array(X_list, dtype=float), np.array(y_list, dtype=int)

    def train_models(
        self,
        train_trace: List[int],
    ) -> Tuple[LogisticRegressionClassifier, DecisionTreeClassifierCustom]:
        """Extract oracle labels and train both Logistic Regression and Decision Tree."""
        X, y = self.generate_training_data_from_oracle(train_trace)
        log_reg = LogisticRegressionClassifier(learning_rate=0.08, n_epochs=70, l2_reg=0.005)
        log_reg.fit(X, y)

        tree = DecisionTreeClassifierCustom(max_depth=4, min_samples_split=6)
        tree.fit(X, y)

        return log_reg, tree

    def run_comparative_simulation(
        self,
        eval_trace: List[int],
        shift_point: int,
        log_reg_model: LogisticRegressionClassifier,
        tree_model: DecisionTreeClassifierCustom,
        record_explanations: bool = True,
    ) -> Dict[str, Any]:
        """Run FIFO, LRU, Optimal, Learned-Logistic, Learned-Tree, and Adaptive-Hybrid.

        Returns comprehensive metrics across Phase 1, Phase 2, and Overall.
        """
        policies = {
            "FIFO": FIFOPolicy(self.capacity),
            "LRU": LRUPolicy(self.capacity),
            "Optimal (Belady)": OptimalPolicy(self.capacity),
            "Learned (Logistic)": LearnedPolicy(self.capacity, log_reg_model, name="Learned (Logistic)"),
            "Learned (Tree)": LearnedPolicy(self.capacity, tree_model, name="Learned (Tree)"),
            "Adaptive-Hybrid": AdaptiveHybridPolicy(self.capacity, log_reg_model, name="Adaptive-Hybrid"),
        }

        # Track cumulative faults per policy
        cumulative_faults: Dict[str, List[int]] = {name: [] for name in policies}
        phase1_hits: Dict[str, int] = {name: 0 for name in policies}
        phase1_faults: Dict[str, int] = {name: 0 for name in policies}
        phase2_hits: Dict[str, int] = {name: 0 for name in policies}
        phase2_faults: Dict[str, int] = {name: 0 for name in policies}

        explanation_records: List[Dict[str, Any]] = []

        # Step-by-step execution
        n_total = len(eval_trace)
        for t, page in enumerate(eval_trace):
            is_phase1 = (t < shift_point)
            future = eval_trace[t + 1 :]

            # Pre-calculate Belady's choice if an eviction occurs for comparison
            optimal_evict = None
            opt_cache = policies["Optimal (Belady)"].cache
            if page not in opt_cache and len(opt_cache) >= self.capacity:
                furthest_dist = -1
                next_pos = {}
                for idx, p in enumerate(future):
                    if p in opt_cache and p not in next_pos:
                        next_pos[p] = idx
                for res in opt_cache:
                    dist = next_pos.get(res, 999999)
                    if dist > furthest_dist:
                        furthest_dist = dist
                        optimal_evict = res

            for name, pol in policies.items():
                # If recording explanation for Learned (Logistic)
                if record_explanations and name == "Learned (Logistic)" and page not in pol.cache and len(pol.cache) >= self.capacity:
                    res_pages = list(pol.cache.keys())
                    feat_dict = {p: pol.extract_features(p, t) for p in res_pages}
                    X_mat = np.array([feat_dict[p] for p in res_pages])
                    scores = pol.model.predict_proba(X_mat)

                    top_page = res_pages[int(np.argmax(scores))]
                    # Generate explanation
                    exp = self.explanation_engine.generate_explanation(
                        evicted_page=top_page,
                        candidates=res_pages,
                        scores=scores,
                        features_dict=feat_dict,
                        optimal_page=optimal_evict if optimal_evict is not None else top_page,
                        timestep=t,
                    )
                    explanation_records.append(exp)

                is_hit, evicted = pol.access(page, t, future_trace=future)

                if is_phase1:
                    if is_hit:
                        phase1_hits[name] += 1
                    else:
                        phase1_faults[name] += 1
                else:
                    if is_hit:
                        phase2_hits[name] += 1
                    else:
                        phase2_faults[name] += 1

                cumulative_faults[name].append(pol.faults)

        # Aggregate summary table
        results_table = []
        for name, pol in policies.items():
            p1_tot = phase1_hits[name] + phase1_faults[name]
            p2_tot = phase2_hits[name] + phase2_faults[name]
            tot = pol.hits + pol.faults

            p1_hr = (phase1_hits[name] / p1_tot) * 100.0 if p1_tot > 0 else 0.0
            p2_hr = (phase2_hits[name] / p2_tot) * 100.0 if p2_tot > 0 else 0.0
            overall_hr = (pol.hits / tot) * 100.0 if tot > 0 else 0.0
            degradation = p1_hr - p2_hr

            results_table.append({
                "Policy": name,
                "Capacity": self.capacity,
                "Phase1_Hits": phase1_hits[name],
                "Phase1_Faults": phase1_faults[name],
                "Phase1_HitRatio_%": round(p1_hr, 2),
                "Phase2_Hits": phase2_hits[name],
                "Phase2_Faults": phase2_faults[name],
                "Phase2_HitRatio_%": round(p2_hr, 2),
                "Total_Hits": pol.hits,
                "Total_Faults": pol.faults,
                "Overall_HitRatio_%": round(overall_hr, 2),
                "Degradation_%": round(degradation, 2),
            })

        # Bonus calibration analysis
        calib_metrics = None
        if explanation_records:
            confs = [e["confidence"] for e in explanation_records]
            corrs = [e["is_correct"] for e in explanation_records]
            calib_metrics = self.explanation_engine.compute_calibration_metrics(confs, corrs)

        hybrid_stats = {
            "fallback_count": policies["Adaptive-Hybrid"].fallback_count,
            "ml_choice_count": policies["Adaptive-Hybrid"].ml_choice_count,
        }

        return {
            "summary_table": results_table,
            "cumulative_faults": cumulative_faults,
            "explanations": explanation_records,
            "calibration_metrics": calib_metrics,
            "hybrid_stats": hybrid_stats,
            "shift_point": shift_point,
            "total_references": n_total,
        }
