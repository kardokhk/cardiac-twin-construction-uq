"""Summarise the 50-subject metrics: per network and structure, median Dice and HD95 with 2.5th-97.5th percentiles,
and the count of hearts meeting the pre-declared usable rule. Writes a Markdown table to stdout."""
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1], sep="\t")
df = df[df.status == "ok"]
print("| network | n | usable all 3 | " + " | ".join(f"{s} Dice median (2.5-97.5%)" for s in ("LV", "MYO", "RV"))
      + " | " + " | ".join(f"{s} HD95 mm median (2.5-97.5%)" for s in ("LV", "MYO", "RV")) + " |")
print("|" + "---|" * 9)
for net, g in df.groupby("network", sort=False):
    cells = [net, str(len(g)), str(int(g.usable_all.astype(str).eq("True").sum()))]
    for m in ("dice", "hd95"):
        for s in ("LV", "MYO", "RV"):
            v = g[f"{s}_{m}"].astype(float)
            fmt = "{:.3f}" if m == "dice" else "{:.1f}"
            cells.append(f"{fmt.format(v.median())} ({fmt.format(v.quantile(0.025))} to {fmt.format(v.quantile(0.975))})")
    print("| " + " | ".join(cells) + " |")
fails = df[~df.usable_all.astype(str).eq("True")]
if len(fails):
    print("\nHearts failing the usable rule:")
    for _, r in fails.iterrows():
        print(f"- {r.network} {r.patient} ({r.group}): Dice LV {r.LV_dice:.3f}, MYO {r.MYO_dice:.3f}, RV {r.RV_dice:.3f}")
