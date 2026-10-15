"""Reproduce every result and figure:   python run_all.py          (full, ~6 min)
                                        python run_all.py --quick  (smoke test, ~1 min)"""
import argparse, csv, json, os, time
from dataclasses import replace

import numpy as np

from uav_bandit import plots
from uav_bandit.experiments import (BLOCK, EnvConfig, experiment_ablation, experiment_battery,
                                    experiment_battery_extras, experiment_long_horizon, experiment_main, experiment_sensitivity,
                                    paired_test, policy_map, tune)

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
Q = args.quick
seeds = list(range(3 if Q else 10)); seedsB = list(range(2 if Q else 5))
epA, epB = (12, 20) if Q else (30, 100)
os.makedirs("results", exist_ok=True); t0 = time.time()
cfgA = EnvConfig(); cfgB = EnvConfig(battery_budget=100.0)

print("[1/6] tuning hyper-parameters on held-out seeds ...")
hp, tuning_log = tune(cfgA, cfgB, epA=epA, epB=max(epB // 2, 10), verbose=False)
print("      chosen:", hp)

print("[2/6] main comparison (memoryless tasks) ...")
out, curves, regrets, oracle, recs = experiment_main(cfgA, hp, seeds, epA)
pm = policy_map(recs)
paired = {n: paired_test(out[n]["cum_regret"]["raw"], out["Always bit"]["cum_regret"]["raw"])
          for n in ("LinUCB", "Thompson Sampling", "Eps-greedy (contextual)", "DQN")}

print("[3/6] context ablation ...")
abl = experiment_ablation(cfgA, hp, seeds, epA)

print("[4/6] sensitivity sweeps ...")
sens = experiment_sensitivity(cfgA, hp, seeds[:5])

print("[5/6] battery-constrained (sequential) setting ...")
bat, bat_curves = experiment_battery(cfgB, hp, seedsB, epB)
gam, budget = experiment_battery_extras(cfgB, hp, seedsB, epB)
long_run = experiment_long_horizon(cfgA, hp, seeds, 30 if Q else 100)

print("[6/6] writing figures + tables ...")
plots.learning_curves(curves, oracle, BLOCK, "results/fig_learning_curves.png")
plots.regret_curves(regrets, BLOCK, "results/fig_regret.png")
plots.comparison_bar(out, oracle, "results/fig_comparison.png")
plots.policy_map(pm, "results/fig_policy_map.png")
plots.ablation(abl, "results/fig_ablation.png")
plots.sensitivity(sens, "results/fig_sensitivity.png")
plots.battery(bat, bat_curves, "results/fig_battery.png")

def strip(d):   # drop raw per-seed lists from the exported summary tables
    return {a: {k: {kk: vv for kk, vv in v.items() if kk != "raw"} for k, v in m.items()} for a, m in d.items()}

summary = {"quick": Q, "hyperparameters": hp, "oracle_avg_reward": oracle, "main": strip(out),
           "curves": curves, "regrets": regrets, "block": BLOCK, "paired_vs_always_bit": paired,
           "policy_map": pm, "ablation": strip(abl), "battery": strip(bat), "battery_curves": bat_curves,
           "battery_gamma_study": strip(gam), "battery_budget_sweep": {b: strip(v) for b, v in budget.items()},
           "sensitivity": {k: {l: strip(v) for l, v in d.items()} for k, d in sens.items()},
           "long_horizon": strip(long_run), "tuning_log": tuning_log, "runtime_sec": time.time() - t0}
json.dump(summary, open("results/summary.json", "w"), indent=1)

with open("results/main_results.csv", "w", newline="") as f:
    w = csv.writer(f); keys = ["avg_reward", "avg_reward_last", "cum_regret", "regret_last", "success_rate",
                               "avg_latency", "avg_energy", "share_bit", "share_semantic", "share_compressed"]
    w.writerow(["agent"] + [f"{k}_{s}" for k in keys for s in ("mean", "std")])
    for n, m in summary["main"].items():
        w.writerow([n] + [round(m[k][s], 4) for k in keys for s in ("mean", "std")])
print(f"done in {time.time() - t0:.0f}s -> results/")
