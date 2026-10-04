"""Synthetic Page-Access Workload Generator with Configurable Distribution Shifts.

Generates realistic traces modeling temporal locality, sequential scans,
and sudden phase shifts to random or bursty thrashing behavior.
"""

from typing import List, Tuple, Dict, Any
import numpy as np


class WorkloadGenerator:
    """Generates synthetic memory reference traces with controlled distribution shifts."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_zipf_trace(self, n_references: int, num_pages: int, alpha: float = 1.2) -> List[int]:
        """Generate a trace following a Zipfian distribution (heavy temporal locality)."""
        ranks = np.arange(1, num_pages + 1)
        probabilities = 1.0 / (ranks ** alpha)
        probabilities /= probabilities.sum()
        return self.rng.choice(num_pages, size=n_references, p=probabilities).tolist()

    def generate_sequential_loop(self, n_references: int, start_page: int, loop_size: int) -> List[int]:
        """Generate repeating sequential scans simulating loop iterations."""
        loop = [start_page + (i % loop_size) for i in range(n_references)]
        return loop

    def generate_uniform_random(self, n_references: int, min_page: int, max_page: int) -> List[int]:
        """Generate uniform random memory references (zero temporal locality)."""
        return self.rng.randint(min_page, max_page, size=n_references).tolist()

    def generate_bursty_trace(self, n_references: int, num_pages: int, burst_interval: int = 50) -> List[int]:
        """Generate references with bursty changes in local working sets."""
        trace = []
        current_focus = self.rng.randint(0, num_pages)
        for i in range(n_references):
            if i % burst_interval == 0:
                current_focus = self.rng.randint(0, num_pages)
            if self.rng.rand() < 0.75:
                # 75% local clustering around focus
                offset = self.rng.choice([-2, -1, 0, 1, 2])
                page = max(0, min(num_pages - 1, current_focus + offset))
            else:
                page = self.rng.randint(0, num_pages)
            trace.append(int(page))
        return trace

    def generate_shifted_workload(
        self,
        total_references: int = 4000,
        num_pages: int = 100,
        shift_ratio: float = 0.5,
        phase1_type: str = "locality_heavy",
        phase2_type: str = "random_bursty",
    ) -> Tuple[List[int], int, Dict[str, Any]]:
        """Generate a complete workload with a deliberate distribution shift at shift_point.

        Args:
            total_references: Total length of the trace.
            num_pages: Size of virtual page space.
            shift_ratio: Fraction of references before the shift (default 0.5).
            phase1_type: 'locality_heavy' (Zipf + sequential loops)
            phase2_type: 'random_bursty' (Uniform random + Poisson working set bursts)

        Returns:
            trace: List of page references.
            shift_point: The index where Phase 2 begins.
            metadata: Description of workload parameters.
        """
        n_phase1 = int(total_references * shift_ratio)
        n_phase2 = total_references - n_phase1
        shift_point = n_phase1

        # Phase 1: High Locality (80% Zipf, 20% sequential loops)
        if phase1_type == "locality_heavy":
            n_zipf = int(n_phase1 * 0.75)
            n_loop = n_phase1 - n_zipf
            part_zipf = self.generate_zipf_trace(n_zipf, num_pages=int(num_pages * 0.4), alpha=1.35)
            part_loop = self.generate_sequential_loop(n_loop, start_page=0, loop_size=12)
            # Interleave naturally
            phase1_trace = []
            iz, il = 0, 0
            for i in range(n_phase1):
                if (i % 5 == 0 or iz >= n_zipf) and il < n_loop:
                    phase1_trace.append(part_loop[il])
                    il += 1
                elif iz < n_zipf:
                    phase1_trace.append(part_zipf[iz])
                    iz += 1
                else:
                    phase1_trace.append(part_loop[il])
                    il += 1
        else:
            phase1_trace = self.generate_uniform_random(n_phase1, 0, num_pages)

        # Phase 2: High Entropy / Shifted (50% Uniform random across all pages, 50% Bursty jumps)
        if phase2_type == "random_bursty":
            n_rand = n_phase2 // 2
            n_burst = n_phase2 - n_rand
            part_rand = self.generate_uniform_random(n_rand, 0, num_pages)
            part_burst = self.generate_bursty_trace(n_burst, num_pages, burst_interval=30)
            phase2_trace = []
            ir, ib = 0, 0
            for i in range(n_phase2):
                if (i % 2 == 0 and ir < n_rand) or ib >= n_burst:
                    phase2_trace.append(part_rand[ir])
                    ir += 1
                else:
                    phase2_trace.append(part_burst[ib])
                    ib += 1
        elif phase2_type == "pure_random":
            phase2_trace = self.generate_uniform_random(n_phase2, 0, num_pages)
        else:
            phase2_trace = self.generate_sequential_loop(n_phase2, start_page=20, loop_size=num_pages)

        full_trace = phase1_trace + phase2_trace
        metadata = {
            "total_references": total_references,
            "shift_point": shift_point,
            "num_pages": num_pages,
            "phase1_type": phase1_type,
            "phase2_type": phase2_type,
        }
        return full_trace, shift_point, metadata
