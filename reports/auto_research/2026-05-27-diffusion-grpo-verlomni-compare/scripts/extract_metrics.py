#!/usr/bin/env python3
"""Extract per-step metrics from a verl-omni or NeMo-RL run log and append to a TSV.

Both stacks log differently:
  - NeMo-RL: lines like `[diffusion_grpo] step=N train/loss=X train/mean_ratio=Y train/reward_mean=Z`
  - verl-omni console: lines like `step:N - actor/pg_loss:X actor/ratio_mean:Y critic/rewards/mean:Z`
    (or via verl.utils.tracking ConsoleTracking → may use dict format)

Usage:
    extract_metrics.py <stack> <log_path> <tsv_path>
    stack: "nemo_rl" or "verl_omni"
"""
import re
import sys
from pathlib import Path


def extract_nemo_rl(log_text: str) -> list[dict]:
    rows = []
    pat = re.compile(
        r"\[diffusion_grpo\] step=(?P<step>\d+) "
        r"train/loss=(?P<loss>[-\d.eE+]+) "
        r"train/mean_ratio=(?P<ratio>[-\d.eE+]+) "
        r"train/reward_mean=(?P<reward>[-\d.eE+]+)"
    )
    for m in pat.finditer(log_text):
        rows.append(
            {
                "step": int(m["step"]),
                "policy_loss": float(m["loss"]),
                "mean_ratio": float(m["ratio"]),
                "clipfrac": "",
                "reward_mean": float(m["reward"]),
            }
        )
    return rows


def extract_verl_omni(log_text: str) -> list[dict]:
    """Parse verl-omni's console-logger output.

    Each training step writes one long single line like:
      step:1 - actor/ppo_kl:... - actor/pg_clipfrac:0.0 - actor/ratio_mean:1.0 -
              actor/loss:1.9e-06 - ... - critic/rewards/mean:-0.010924 - ...
    """
    rows = []
    line_pat = re.compile(r"\bstep:(\d+)\s+-\s+actor/ppo_kl:")
    for m in line_pat.finditer(log_text):
        step = int(m.group(1))
        # Capture from this `step:` marker up to the next `step:` (or end).
        start = m.start()
        next_m = line_pat.search(log_text, start + 1)
        body = log_text[start : next_m.start() if next_m else len(log_text)]

        def _grab(key: str) -> str:
            mm = re.search(rf"{re.escape(key)}:([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan)", body)
            return mm.group(1) if mm else ""

        rows.append(
            {
                "step": step,
                "policy_loss": _grab("actor/loss"),
                "mean_ratio": _grab("actor/ratio_mean"),
                "clipfrac": _grab("actor/pg_clipfrac"),
                "reward_mean": _grab("critic/rewards/mean"),
            }
        )
    return rows


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 1
    stack, log_path, tsv_path = sys.argv[1], sys.argv[2], sys.argv[3]
    log_text = Path(log_path).read_text(errors="replace")
    if stack == "nemo_rl":
        rows = extract_nemo_rl(log_text)
    elif stack == "verl_omni":
        rows = extract_verl_omni(log_text)
    else:
        print(f"unknown stack: {stack}", file=sys.stderr)
        return 2
    if not rows:
        print(f"WARN: extracted 0 rows from {log_path}", file=sys.stderr)
    tsv = Path(tsv_path)
    new_file = not tsv.exists()
    with tsv.open("a") as f:
        if new_file:
            f.write(
                "stack\tstep\tpolicy_loss\tmean_ratio\tclipfrac\treward_mean\tlog\n"
            )
        for r in rows:
            f.write(
                f"{stack}\t{r['step']}\t{r['policy_loss']}\t{r['mean_ratio']}\t"
                f"{r['clipfrac']}\t{r['reward_mean']}\t{log_path}\n"
            )
    print(f"appended {len(rows)} rows to {tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
