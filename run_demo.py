#!/usr/bin/env python3
"""Application-level check on real text messages (examples/messages.txt).
   1) decision grid: every message x several channel qualities
   2) message-level comparison of policies over random channel draws
Writes results/demo_decisions.csv, results/demo_summary.json, results/fig_demo_grid.png"""
import csv, json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from uav_bandit.environment import MODES, EnvConfig, Situation
from uav_bandit.messaging import URGENCY_NAMES, analyze_message, transmit
from uav_bandit.policy import load_policy, oracle_rewards, select_mode

msgs = [l.strip() for l in open("examples/messages.txt", encoding="utf-8") if l.strip()]
SNRS = [0, 10, 20, 28]
agent, cfg = load_policy(), EnvConfig()
profiles = [analyze_message(t) for t in msgs]
sit_of = lambda p, snr: Situation(snr, 0, p.urgency, p.deadline, p.min_quality)

# ---- 1) decision grid
rows = []
for i, (t, p) in enumerate(zip(msgs, profiles)):
    for snr in SNRS:
        s = sit_of(p, snr)
        m = select_mode("linucb", agent, s); o = int(np.argmax(oracle_rewards(s, cfg)))
        r = transmit(p, m, snr, np.random.default_rng(1000 * i + snr), cfg)
        rows.append(dict(msg_id=i + 1, message=t, type=p.task_type, urgency=URGENCY_NAMES[p.urgency], deadline=p.deadline,
                         snr=snr, chosen=MODES[m], oracle=MODES[o], payload_bytes=r.payload, original_bytes=len(t.encode()),
                         quality=round(r.quality, 2), success=r.success, reward=round(r.reward, 2)))
with open("results/demo_decisions.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# ---- 2) message-level policy comparison (same channel draws for every policy)
R = 300
def evaluate(policy):
    rew, suc, en, by = [], [], [], []
    for i, p in enumerate(profiles):
        rng_ch = np.random.default_rng(500 + i); rng_tx = np.random.default_rng(900 + i)
        for _ in range(R):
            snr = float(rng_ch.uniform(15, 30) if rng_ch.random() < cfg.los_prob else rng_ch.uniform(-5, 15))
            s = sit_of(p, snr)
            m = int(np.argmax(oracle_rewards(s, cfg))) if policy == "oracle" else select_mode(policy, agent, s)
            r = transmit(p, m, snr, rng_tx, cfg)
            rew.append(r.reward); suc.append(r.success); en.append(r.energy); by.append(r.payload)
    return dict(avg_reward=float(np.mean(rew)), success_rate=float(np.mean(suc)), avg_energy=float(np.mean(en)), avg_payload_bytes=float(np.mean(by)))
summary = {"n_messages": len(msgs), "draws_per_message": R,
           "policies": {p: evaluate(p) for p in ["oracle", "linucb", "bit", "semantic", "compressed", "threshold"]}}
agree = np.mean([r["chosen"] == r["oracle"] for r in rows])
summary["grid_agreement_with_oracle"] = float(agree)
summary["grid_success_rate"] = float(np.mean([r["success"] for r in rows]))
summary["mode_counts_grid"] = {m: int(sum(r["chosen"] == m for r in rows)) for m in MODES}
json.dump(summary, open("results/demo_summary.json", "w"), indent=1)

# ---- figure
col = {"bit": "#1F3A5F", "semantic": "#E0A458", "compressed_semantic": "#0E7C7B"}
fig, ax = plt.subplots(figsize=(9, 7))
for r in rows:
    x, y = SNRS.index(r["snr"]), r["msg_id"] - 1
    ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, color=col[r["chosen"]], alpha=.9))
    ax.text(x, y, "ok" if r["success"] else "fail", ha="center", va="center", color="white", fontsize=8)
ax.set_xlim(-.5, len(SNRS) - .5); ax.set_ylim(len(msgs) - .5, -.5)
ax.set_xticks(range(len(SNRS))); ax.set_xticklabels([f"{s} dB" for s in SNRS])
ax.set_yticks(range(len(msgs))); ax.set_yticklabels([f"{p.task_type[:7]}/{URGENCY_NAMES[p.urgency][:4]}  {t[:34]}..." for t, p in zip(msgs, profiles)], fontsize=7)
ax.set_title("Mode chosen by the learned policy for each message and channel quality"); ax.set_xlabel("Channel SNR")
ax.legend(handles=[Patch(color=c, label=m) for m, c in col.items()], loc="upper center", bbox_to_anchor=(.5, -.08), ncol=3, frameon=False)
fig.tight_layout(); fig.savefig("results/fig_demo_grid.png", dpi=160)

print("Message-level comparison (%d messages x %d random channel draws):" % (len(msgs), R))
for p, v in summary["policies"].items():
    print(f"  {p:11s} reward {v['avg_reward']:+.3f}   success {v['success_rate']*100:5.1f}%   energy {v['avg_energy']:.2f}   payload {v['avg_payload_bytes']:.0f} B")
print(f"Grid: policy matches oracle mode in {agree*100:.0f}% of cells; modes used: {summary['mode_counts_grid']}")
