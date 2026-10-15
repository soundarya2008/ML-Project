#!/usr/bin/env python3
"""Type a message; the system analyses it, picks bit / semantic / compressed-semantic, and 'sends' it.

  python send_message.py "URGENT: obstacle ahead, reduce altitude to 40 meters" --snr 12
  python send_message.py --interactive --snr random
  python send_message.py --file examples/messages.txt --snr 8 --compare
"""
import argparse
import sys

import numpy as np

from uav_bandit.environment import MODES, EnvConfig, Situation
from uav_bandit.messaging import (URGENCY_NAMES, analyze_message, explain_modes, payload_bytes, transmit)
from uav_bandit.policy import (POLICIES, estimate_rewards, load_policy, oracle_rewards, select_mode)

NAMES = ("BIT", "SEMANTIC", "COMPRESSED SEMANTIC")


def draw_snr(rng, cfg):
    return float(rng.uniform(15, 30) if rng.random() < cfg.los_prob else rng.uniform(-5, 15))


def report(text, snr, args, agent, cfg, rng):
    prof = analyze_message(text, args.type, None if args.urgency is None else URGENCY_NAMES.index(args.urgency), args.deadline)
    sit = Situation(snr, 0, prof.urgency, prof.deadline, prof.min_quality)
    mode = select_mode(args.policy, agent, sit)
    est = estimate_rewards(agent, sit)
    truth = oracle_rewards(sit, cfg)
    line = "=" * 74
    print(f"\n{line}\nMESSAGE: {text}\n{line}")
    print("1. ANALYSIS")
    print(f"   type: {prof.task_type:16s} urgency: {URGENCY_NAMES[prof.urgency]:9s} deadline: {prof.deadline:.1f} s   required quality: {prof.min_quality:.2f}")
    print(f"   key units ({len(prof.key_units)}): " + " ".join(u[0] for u in prof.key_units))
    print(f"2. CHANNEL   SNR = {snr:.1f} dB ({'clear line-of-sight' if snr >= 15 else 'blocked / weak'})")
    print(f"3. DECISION  policy = {args.policy}")
    print("   estimated reward  " + "   ".join(f"{MODES[i]}: {est[i]:+.2f}" for i in range(3)) if args.policy == "linucb" else "   (fixed rule, no learned estimate)")
    for e in explain_modes(prof, snr):
        print("     - " + e)
    print(f"   >>> CHOSEN MODE: {NAMES[mode]}   (best possible in expectation: {NAMES[int(np.argmax(truth))]})")
    res = transmit(prof, mode, snr, rng, cfg)
    print("4. TRANSMISSION")
    print(f"   payload {res.payload} bytes (original {payload_bytes(prof, 0)} bytes)   latency {res.latency:.2f} s   energy {res.energy:.2f}")
    print(f"   {res.note}")
    print(f"   receiver got: \"{res.received_text}\"" if res.received_text else "   receiver got: nothing")
    print(f"5. OUTCOME   quality {res.quality:.2f} (needs {prof.min_quality:.2f})  deadline {'met' if res.deadline_met else 'MISSED'}  "
          f"-> {'SUCCESS' if res.success else 'FAILED'}   reward {res.reward:+.2f}")
    if args.compare:
        print("   what each mode would have done in this same channel:")
        for m in range(3):
            r = transmit(prof, m, snr, np.random.default_rng(args.seed + m), cfg)
            print(f"     {NAMES[m]:20s} payload {r.payload:4d} B  quality {r.quality:.2f}  {'success' if r.success else 'FAILED '}  reward {r.reward:+.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("message", nargs="?", help="the message to send")
    ap.add_argument("--snr", default="random", help="channel SNR in dB, or 'random' (default)")
    ap.add_argument("--policy", default="linucb", choices=POLICIES)
    ap.add_argument("--type", choices=["image", "sensor_reading", "control_command"], help="override detected type")
    ap.add_argument("--urgency", choices=list(URGENCY_NAMES), help="override detected urgency")
    ap.add_argument("--deadline", type=float, help="override deadline in seconds")
    ap.add_argument("--interactive", action="store_true", help="keep asking for messages")
    ap.add_argument("--file", help="send every line of a text file")
    ap.add_argument("--compare", action="store_true", help="also show what the other two modes would have done")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg, rng = EnvConfig(), np.random.default_rng(args.seed)
    agent = load_policy()

    def snr_value():
        return draw_snr(rng, cfg) if args.snr == "random" else float(args.snr)

    if args.file:
        for line in open(args.file, encoding="utf-8"):
            if line.strip(): report(line.strip(), snr_value(), args, agent, cfg, rng)
    elif args.interactive:
        print("Type a message and press Enter (empty line to quit).")
        while True:
            try: text = input("\nmessage> ").strip()
            except EOFError: break
            if not text: break
            report(text, snr_value(), args, agent, cfg, rng)
    elif args.message:
        report(args.message, snr_value(), args, agent, cfg, rng)
    else:
        ap.print_help(); sys.exit(1)


if __name__ == "__main__":
    main()
