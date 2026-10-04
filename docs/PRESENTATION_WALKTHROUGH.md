# In-Class Code Walkthrough & Demo Guide (3–5 Minutes)
**Course:** CSE-307: Operating Systems (Term Paper — Section B)  
**Topic:** Learning-Augmented OS Heuristics: Classical Algorithms Meet Adaptive Prediction  
**Track:** Track 1 (Learned Page Replacement) + Bonus Track (Explanation Confidence & Calibration)

---

## 1. Quick Presentation Overview

* **Target Duration:** 3 minutes 30 seconds to 4 minutes 30 seconds (Leaves 30–60 seconds for Q&A).
* **Objective:** Clearly demonstrate:
  1. Correct implementation of classical OS algorithms (FIFO, LRU, Belady MIN).
  2. The learning-augmented model (features, inference, and adaptive fallback).
  3. The experimental shift scenario and the critical finding: **which policy degrades the most and why**.
  4. The bonus track: natural-language eviction explanations and confidence calibration (ECE).

---

## 2. Minute-by-Minute Speaking Script

### Minute 0:00 – 0:45 | Problem Motivation & Architecture
> *"Good morning everyone. Today I'm presenting our term paper on Learning-Augmented OS Heuristics, focusing on Page Replacement in Virtual Memory.*  
> *Classical operating systems rely on fixed heuristics like FIFO and LRU. Belady's MIN algorithm is the offline optimal benchmark, but it requires knowing the future. While LRU works well under stable temporal locality, real-world systems constantly face sudden workload shifts—such as transitioning from tight application loops to broad database scans or randomized memory access.*  
> *We set out to investigate: Can a lightweight machine learning classifier predict good eviction candidates, and what happens when the workload abruptly shifts partway through? Furthermore, can an adaptive hybrid policy protect the OS from catastrophic failure?"*

### Minute 0:45 – 1:45 | System Implementation & Live Demo
*(Switch to terminal and run the test suite / reproduction command)*
```bash
python3 -m unittest discover tests
```
> *"To ensure academic rigor, we first validated all classical algorithms against textbook canonical reference traces—confirming exact fault counts for FIFO (15 faults), LRU (12 faults), and Belady's Optimal (9 faults).*  
> *On top of these baselines, we extracted four real-time telemetry features for every resident page: recency, access frequency, cache residence age, and previous reuse distance. We trained lightweight pure-NumPy Logistic Regression and Decision Tree models on Belady oracle feedback from a warm-up trace.*  
> *Crucially, we implemented an **Adaptive-Hybrid** policy that monitors the classifier's prediction confidence margin. If the model's confidence drops below a threshold, or if sliding-window miss rates spike, it safely falls back to LRU."*

### Minute 1:45 – 3:15 | The Stress Test: Workload Shift & Core Findings
*(Show Figure 1 and Figure 2 or display the markdown summary table)*
```bash
cat results/performance_table.md
```
> *"Now for the core experiment: We generated a 6,000-access reference trace with a deliberate, sharp distribution shift at tick 3,000. Phase 1 is locality-heavy (Zipfian plus sequential loops). Phase 2 suddenly switches to high-entropy uniform random access with bursty working-set sweeps.*  
> *Here is the key question from the brief: **Which policy degrades the most and why?***  
> *As shown in Table 1 and Figure 1:*  
> *1. **Pure Machine Learning Degrades the Most:** The standalone Learned Logistic model collapsed by **65.63 percentage points** (falling from 80.17% down to 14.53%), and the Decision Tree collapsed by **62.63 percentage points**. In Phase 2, pure ML incurred over 2,500 faults, performing worse than half the efficiency of naive FIFO!*  
> *Why? The ML model learned statistical patterns of recency and frequency that were valid only during Phase 1. When the distribution became uniform random, past frequency became uninformative. Worse, the model suffered from **pathological retention bias**—it held onto obsolete pages with high historical counts, creating 'zombie frames' that polluted the cache.*  
> *2. **Classical Heuristics Degrade Gracefully:** LRU degraded by 46.90% and FIFO by 45.47%. Because LRU continuously flushes the tail of the cache regardless of past frequency, it purged stale items much faster than the confused ML model.*  
> *3. **The Hybrid Solution:** Look at the Adaptive-Hybrid row: it achieved 82.83% in Phase 1, and when the shift hit, its confidence monitor detected the anomaly and switched to LRU fallback, matching LRU's 35.93% and completely preventing the 65% collapse of pure ML."*

### Minute 3:15 – 4:15 | Bonus Track: Explanations & Calibration (+10 Pts)
*(Show Figure 4 and sample explanations)*
```bash
head -n 20 results/sample_explanations.txt
```
> *"For the bonus track, we implemented an Eviction Explanation Engine. At every eviction, the system generates a natural-language rationale and a self-rated confidence score between 0.50 and 0.99.*  
> *Across 3,143 eviction events, our empirical agreement with Belady was 66.94%, while mean self-rated confidence was 82.30%.*  
> *We evaluated whether the confidence score was well-calibrated using a Reliability Diagram and Expected Calibration Error (ECE = 0.1736). As you see in Figure 4, the model has strong relative calibration: decisions with confidence over 90% achieved greater than 78% correctness, whereas low-confidence decisions hovered around 54%. However, an overconfidence gap opened up post-shift because linear models lack epistemic uncertainty awareness under out-of-distribution data."*

### Minute 4:15 – 4:45 | Conclusion & Takeaways
> *"To conclude, our study proves three vital lessons for modern OS design:*  
> *First, unconstrained machine learning in the OS kernel is dangerous because distribution shifts cause catastrophic thrashing.*  
> *Second, learning-augmented systems MUST include safety fallbacks to bounded classical algorithms like LRU.*  
> *Third, calibrated uncertainty estimation is essential so the OS knows when its predictor cannot be trusted.*  
> *Thank you, and I welcome any questions!"*

---

## 3. Live Demo Cheat Sheet (Terminal Commands)

During your in-class demonstration, keep your terminal in the repository directory and run these commands:

| Step | Command | What to Explain to Class |
| :--- | :--- | :--- |
| **1. Verify Tests** | `python3 -m unittest discover tests` | Shows all 4 unit tests passing, verifying canonical textbook traces. |
| **2. Run Benchmark** | `python3 experiments/run_experiments.py` | Runs simulation across all 6 policies, prints model weights and summary table. |
| **3. View Results** | `cat results/performance_table.md` | Points to the Degradation column highlighting the 65% drop in pure ML. |
| **4. Show Explanations** | `head -n 15 results/sample_explanations.txt` | Displays the bonus track natural language explanations and confidence scores. |
| **5. Open Figures** | `xdg-open results/figure1_hit_ratios.png` | Displays the bar chart comparing pre-shift vs post-shift hit ratios. |

---

## 4. Anticipated Questions from Professor/TA & Model Answers

### Q1: "Why did you implement the ML models from scratch in NumPy rather than using scikit-learn?"
> **Answer:**  
> *"Inside an OS kernel or low-level runtime, external heavy dependencies and large C-extensions are non-starters due to memory footprint and determinism constraints. Implementing Logistic Regression and Decision Trees directly in pure NumPy gives us sub-microsecond inference latency, deterministic execution, and full inspectability into feature weights."*

### Q2: "Why did FIFO outperform pure ML in Phase 2?"
> **Answer:**  
> *"In Phase 2, accesses are uniformly distributed with no temporal locality. FIFO simply rotates pages in a ring buffer, meaning every resident page has an equal residence lifetime. In contrast, the trained ML model has learned positive weights for certain frequency/age combinations; it continuously holds onto 'zombie pages' from Phase 1 that never get accessed again, reducing the effective cache size and causing massive thrashing."*

### Q3: "How does the Adaptive-Hybrid policy detect that a shift occurred?"
> **Answer:**  
> *"It uses two complementary signals: First, an intra-decision confidence margin (the difference between the top eviction candidate's score and the second candidate). When candidate features become erratic, margins shrink. Second, a sliding-window fault rate monitor over the last 50 accesses. When the hit rate drops below 15%, the policy flags a phase shift anomaly and immediately routes eviction decisions through LRU."*

### Q4: "What does the Expected Calibration Error (ECE = 0.1736) tell us about the bonus track?"
> **Answer:**  
> *"An ECE of 0.1736 indicates that while the model is monotonic (higher confidence generally correlates with higher accuracy), it is systematically overconfident by roughly 17%. This occurs because the model's confidence was derived from sigmoid activation margins trained on stationary data; when the distribution shifted out-of-distribution, it had no mechanism to express epistemic uncertainty."*
