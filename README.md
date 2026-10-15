# Contextual Bandit Hybrid Bit-Semantic Transmission Selection for UAVs

**Machine Learning (25SC2107E) — Term End Project & Hackathon Submission**
KLH University (Koneru Lakshmaiah Education Foundation)

| | |
|---|---|
| **Team** | Likhita Sri Chakiri (2520040067) &middot; Satwika Gunda (2520040051) &middot; Soundarya Valli J (2520040043) |
| **Guide** | Mr. N Lava Prasad, Assistant Professor, Department of ECE |
| **A.Y.** | 2026-27 |

---

## What this project does

A UAV must decide, for every message it needs to send, whether to transmit **raw bits**, the **semantic
meaning**, or a **compressed semantic** version. The right choice depends on the current channel quality,
the message's urgency, its delay budget, and how much battery is left — so instead of hard-coding a rule,
this project **learns the decision online** using contextual multi-armed bandits (LinUCB, linear Thompson
Sampling), a contextual epsilon-greedy baseline, and a Deep Q-Network implemented from scratch in pure
NumPy, and compares all of them against an oracle upper bound and several non-learning baselines.

**Headline result:** on 6,000 simulated tasks over 10 random seeds, LinUCB cuts cumulative regret against
the best non-learning baseline (always send raw bits) by **27%**, consistently across every seed tested
(paired t = -3.83), and agrees with an oracle's transmission-mode choice on **76.7%** of tasks.

## Repository contents

This repository contains every required submission artifact for the Term End Project Defense and
Hackathon evaluation:

| Folder | Contents |
|---|---|
| [`00_Supplementary_Materials/`](./00_Supplementary_Materials) | Problem statement document, project guide, and oral presentation script |
| [`01_Abstract/`](./01_Abstract) | Project abstract |
| [`02_Literature_Survey/`](./02_Literature_Survey) | Review of 9 reference papers (semantic communication, UAV task-oriented comms, contextual bandits) with relevance to this project |
| [`03_Presentations/`](./03_Presentations) | Project review slide deck(s) |
| [`04_Implementation_Code/`](./04_Implementation_Code) | Full Python implementation — environment, agents (LinUCB/Thompson Sampling/DQN), experiments, plotting, CLI tools, tests |
| [`05_Final_Report/`](./05_Final_Report) | Final project report (22 pages), built on the official KLH report template |
| [`06_Dataset_and_EDA/`](./06_Dataset_and_EDA) | Dataset & EDA note — this is a simulation-based RL project with no external dataset, so this explains and statistically profiles the simulated task/channel distribution used for training and evaluation |
| [`results/`](./results) | All 8 result figures (PNG) and the raw experiment summaries (JSON) referenced in the report |

## Quick start (run the code yourself)

```bash
cd 04_Implementation_Code
pip install -r requirements.txt
python -m pytest -q tests          # 9 sanity tests
python run_all.py --quick          # smoke test, 1-2 min
python run_all.py                  # full reproduction, ~8.7 min, regenerates results/
```

No GPU and no deep-learning framework is required — every learning algorithm, including the DQN, is
implemented with plain NumPy.

```bash
python send_message.py "URGENT: obstacle detected ahead, reduce altitude to 40 meters immediately" --snr 12 --compare
python run_demo.py                 # 16 real messages x 4 channel qualities + policy comparison
```

## Method summary

- **Context (13 features):** channel SNR, task urgency (one-hot), delay budget, minimum quality
  requirement, battery fraction, plus 3 engineered "margin" features that express how far the channel is
  from each mode's success threshold.
- **Learners:** LinUCB and linear Thompson Sampling (Sherman-Morrison incremental updates, implemented
  from scratch), contextual epsilon-greedy, and a from-scratch NumPy DQN.
- **Baselines:** always-bit, always-semantic, always-compressed, random, a hand-written SNR threshold rule.
- **Oracle:** best achievable expected reward per task — used only as an upper bound, never as a trainable
  policy.
- **Protocol:** every hyperparameter tuned on held-out seeds disjoint from the 10 evaluation seeds, to
  avoid tuning on the test data.

## Key results (Setting A: 6,000 independent tasks, 10 seeds; oracle = 0.696)

| Policy | Avg. reward | Regret | Success % | Energy |
|---|---|---|---|---|
| Always bit | 0.423 ± 0.053 | 1642 | 76.5 | 1.00 |
| Threshold rule | -0.822 ± 0.030 | 9208 | 58.4 | 0.79 |
| Eps-greedy (contextual) | 0.442 ± 0.081 | 1611 | 75.5 | 0.92 |
| **LinUCB** | **0.507 ± 0.053** | **1206** | 76.0 | 0.91 |
| Thompson Sampling | 0.503 ± 0.073 | 1299 | 76.1 | 0.92 |
| DQN | -0.082 ± 0.049 | 4652 | 66.8 | 0.77 |

Full results, ablations, sensitivity sweeps, the battery-constrained setting, and the real-message
demonstration are in [`05_Final_Report/`](./05_Final_Report).

## A note on honesty

In the battery-constrained sequential setting, the DQN agent gets the best mean reward of all eight
policies — but whether that comes from genuine multi-step planning or simply a more conservative energy
policy is **not fully resolved** by this project's experiments, and is reported explicitly as an open
question (see the Limitations section of the final report) rather than claimed as a clean win.
