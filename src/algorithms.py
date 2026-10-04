"""Page Replacement Algorithms: FIFO, LRU, Optimal (Belady), Learned, and Adaptive Hybrid.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
import numpy as np

from src.models import LogisticRegressionClassifier, DecisionTreeClassifierCustom


class BasePageReplacementPolicy(ABC):
    """Abstract Base Class for Page Replacement Policies."""

    def __init__(self, capacity: int, name: str):
        self.capacity = capacity
        self.name = name
        self.hits = 0
        self.faults = 0
        self.cache: Dict[int, Dict[str, Any]] = {}
        self.evictions: List[Tuple[int, int]] = []  # (timestep, evicted_page)
        self.history: List[Dict[str, Any]] = []

    def reset(self):
        self.hits = 0
        self.faults = 0
        self.cache.clear()
        self.evictions.clear()
        self.history.clear()

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.faults
        return (self.hits / total) if total > 0 else 0.0

    @abstractmethod
    def access(
        self,
        page: int,
        timestep: int,
        future_trace: Optional[List[int]] = None,
    ) -> Tuple[bool, Optional[int]]:
        """Access a page at a given timestep.

        Returns:
            (is_hit, evicted_page_or_None)
        """
        pass


class FIFOPolicy(BasePageReplacementPolicy):
    """First-In, First-Out (FIFO) Page Replacement."""

    def __init__(self, capacity: int):
        super().__init__(capacity, "FIFO")
        self.queue: deque = deque()

    def reset(self):
        super().reset()
        self.queue.clear()

    def access(
        self,
        page: int,
        timestep: int,
        future_trace: Optional[List[int]] = None,
    ) -> Tuple[bool, Optional[int]]:
        if page in self.cache:
            self.hits += 1
            return True, None

        self.faults += 1
        evicted = None

        if len(self.cache) >= self.capacity:
            evicted = self.queue.popleft()
            del self.cache[evicted]
            self.evictions.append((timestep, evicted))

        self.queue.append(page)
        self.cache[page] = {"load_time": timestep}
        return False, evicted


class LRUPolicy(BasePageReplacementPolicy):
    """Least Recently Used (LRU) Page Replacement."""

    def __init__(self, capacity: int):
        super().__init__(capacity, "LRU")

    def access(
        self,
        page: int,
        timestep: int,
        future_trace: Optional[List[int]] = None,
    ) -> Tuple[bool, Optional[int]]:
        if page in self.cache:
            self.hits += 1
            self.cache[page]["last_access"] = timestep
            self.cache[page]["access_count"] += 1
            return True, None

        self.faults += 1
        evicted = None

        if len(self.cache) >= self.capacity:
            # Find page with minimal last_access
            evicted = min(self.cache.keys(), key=lambda p: self.cache[p]["last_access"])
            del self.cache[evicted]
            self.evictions.append((timestep, evicted))

        self.cache[page] = {
            "load_time": timestep,
            "last_access": timestep,
            "access_count": 1,
        }
        return False, evicted


class OptimalPolicy(BasePageReplacementPolicy):
    """Belady's Optimal (MIN) Algorithm (Theoretical Upper Bound).

    Evicts the resident page that will not be used for the longest period of time.
    Requires future knowledge of the reference trace.
    """

    def __init__(self, capacity: int):
        super().__init__(capacity, "Optimal (Belady)")

    def access(
        self,
        page: int,
        timestep: int,
        future_trace: Optional[List[int]] = None,
    ) -> Tuple[bool, Optional[int]]:
        if page in self.cache:
            self.hits += 1
            return True, None

        self.faults += 1
        evicted = None

        if len(self.cache) >= self.capacity:
            if future_trace is None:
                raise ValueError("Optimal policy requires future_trace.")

            # Calculate next access distance for all resident pages
            furthest_distance = -1
            candidate = None

            # Pre-compute positions for fast lookup
            next_positions = {}
            for idx, p in enumerate(future_trace):
                if p in self.cache and p not in next_positions:
                    next_positions[p] = idx

            for resident in self.cache:
                if resident not in next_positions:
                    # Page will never be referenced again, prime eviction candidate
                    candidate = resident
                    break
                dist = next_positions[resident]
                if dist > furthest_distance:
                    furthest_distance = dist
                    candidate = resident

            evicted = candidate
            del self.cache[evicted]
            self.evictions.append((timestep, evicted))

        self.cache[page] = {"load_time": timestep}
        return False, evicted


class LearnedPolicy(BasePageReplacementPolicy):
    """Learned Page Replacement using a trained predictive model.

    Extracts features for resident pages and predicts the eviction score
    (higher score = higher probability of long reuse distance / best candidate).
    """

    def __init__(self, capacity: int, model: Any, name: str = "Learned-ML"):
        super().__init__(capacity, name)
        self.model = model
        self.last_explanations: List[Dict[str, Any]] = []

    def extract_features(self, page: int, timestep: int) -> np.ndarray:
        """Features:
        0: Recency (timestep - last_access)
        1: Frequency (access_count)
        2: Age in cache (timestep - load_time)
        3: Previous reuse distance (last_access - prev_access if available, else age)
        """
        meta = self.cache[page]
        recency = float(timestep - meta["last_access"])
        frequency = float(meta["access_count"])
        age = float(timestep - meta["load_time"])
        prev_dist = float(meta.get("prev_reuse_dist", recency))
        return np.array([recency, frequency, age, prev_dist])

    def access(
        self,
        page: int,
        timestep: int,
        future_trace: Optional[List[int]] = None,
    ) -> Tuple[bool, Optional[int]]:
        if page in self.cache:
            self.hits += 1
            meta = self.cache[page]
            meta["prev_reuse_dist"] = timestep - meta["last_access"]
            meta["last_access"] = timestep
            meta["access_count"] += 1
            return True, None

        self.faults += 1
        evicted = None

        if len(self.cache) >= self.capacity:
            resident_pages = list(self.cache.keys())
            feature_matrix = np.array([self.extract_features(p, timestep) for p in resident_pages])

            # Scores: probability/score that page is optimal to evict
            if hasattr(self.model, "predict_proba"):
                scores = self.model.predict_proba(feature_matrix)
            else:
                scores = self.model.predict(feature_matrix)

            best_idx = int(np.argmax(scores))
            evicted = resident_pages[best_idx]

            del self.cache[evicted]
            self.evictions.append((timestep, evicted))

        self.cache[page] = {
            "load_time": timestep,
            "last_access": timestep,
            "access_count": 1,
            "prev_reuse_dist": 1.0,
        }
        return False, evicted


class AdaptiveHybridPolicy(BasePageReplacementPolicy):
    """Learning-Augmented Hybrid Policy with LRU Safety Net.

    Evaluates model prediction confidence. If confidence is below threshold,
    or if sliding window miss-rate suggests model degradation (e.g. during a phase shift),
    gracefully blends or falls back to LRU.
    """

    def __init__(
        self,
        capacity: int,
        model: Any,
        confidence_threshold: float = 0.58,
        sliding_window_size: int = 50,
        name: str = "Adaptive-Hybrid",
    ):
        super().__init__(capacity, name)
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.sliding_window = deque(maxlen=sliding_window_size)
        self.fallback_count = 0
        self.ml_choice_count = 0

    def reset(self):
        super().reset()
        self.sliding_window.clear()
        self.fallback_count = 0
        self.ml_choice_count = 0

    def extract_features(self, page: int, timestep: int) -> np.ndarray:
        meta = self.cache[page]
        recency = float(timestep - meta["last_access"])
        frequency = float(meta["access_count"])
        age = float(timestep - meta["load_time"])
        prev_dist = float(meta.get("prev_reuse_dist", recency))
        return np.array([recency, frequency, age, prev_dist])

    def access(
        self,
        page: int,
        timestep: int,
        future_trace: Optional[List[int]] = None,
    ) -> Tuple[bool, Optional[int]]:
        if page in self.cache:
            self.hits += 1
            self.sliding_window.append(1)
            meta = self.cache[page]
            meta["prev_reuse_dist"] = timestep - meta["last_access"]
            meta["last_access"] = timestep
            meta["access_count"] += 1
            return True, None

        self.faults += 1
        self.sliding_window.append(0)
        evicted = None

        if len(self.cache) >= self.capacity:
            resident_pages = list(self.cache.keys())
            feature_matrix = np.array([self.extract_features(p, timestep) for p in resident_pages])

            scores = self.model.predict_proba(feature_matrix)

            sorted_indices = np.argsort(scores)[::-1]
            top_score = scores[sorted_indices[0]]
            second_score = scores[sorted_indices[1]] if len(scores) > 1 else 0.0
            margin = top_score - second_score

            # Fallback condition: Low model confidence margin OR high recent miss rate (>85%)
            recent_hit_rate = (sum(self.sliding_window) / len(self.sliding_window)) if self.sliding_window else 0.5
            use_fallback = (margin < self.confidence_threshold - 0.5) or (len(self.sliding_window) >= 30 and recent_hit_rate < 0.15)

            if use_fallback:
                # Fallback to LRU
                evicted = min(self.cache.keys(), key=lambda p: self.cache[p]["last_access"])
                self.fallback_count += 1
            else:
                evicted = resident_pages[sorted_indices[0]]
                self.ml_choice_count += 1

            del self.cache[evicted]
            self.evictions.append((timestep, evicted))

        self.cache[page] = {
            "load_time": timestep,
            "last_access": timestep,
            "access_count": 1,
            "prev_reuse_dist": 1.0,
        }
        return False, evicted
