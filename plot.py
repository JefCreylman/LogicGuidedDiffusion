import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FILE_PATH = r"metrics\guidedDiffusionMetrics.csv"

def is125Series(val):
    if val <= 0:
        return False
    leading_digit = np.round(val / (10 ** np.floor(np.log10(val))), 1)
    return leading_digit in [1.0, 2.0, 5.0]

df = pd.read_csv(FILE_PATH, header=None, names=["X", "Y1", "Y2", "Y3"])

baseline = df[df["X"] == 0]
df = df[df["X"] > 0].sort_values(by="X")

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12), sharex=True)

axes[0].plot(df["X"], df["Y1"], marker="o", color="royalblue", linewidth=2)
if not baseline.empty:
    axes[0].axhline(y=baseline["Y1"].values[0], color="royalblue", linestyle='--', alpha = 0.7, label = 'Omega = 0')
    axes[0].legend(loc="upper right")
    ticks = sorted(list(axes[0].get_yticks()) + [baseline["Y1"].values[0]])
    axes[0].set_yticks(ticks, labels=[str(int(x)) if float(x).is_integer() else f"{x:.2f}" for x in ticks])
axes[0].set_ylabel("LeNet FID", fontsize=10)
axes[0].set_title("FID vs. full dataset", fontsize=12, fontweight="bold")
axes[0].set_ylim(0,16)
axes[0].grid(True, which="both", linestyle="--", alpha=0.5)

axes[1].plot(df["X"], df["Y2"], marker="s", color="darkorange", linewidth=2)
# if not baseline.empty:
#     axes[1].axhline(y=baseline["Y2"].values[0], color="royalblue", linestyle='--', alpha = 0.7, label = 'Omega = 0')
#     axes[1].legend(loc="upper right")
#     ticks = sorted(list(axes[1].get_yticks()) + [baseline["Y2"].values[0]])
#     axes[1].set_yticks(ticks, labels=[str(int(x)) if float(x).is_integer() else f"{x:.2f}" for x in ticks])
axes[1].set_ylabel("LeNet FID", fontsize=10)
axes[1].set_title("FID vs. target dataset", fontsize=12, fontweight="bold")
axes[1].set_ylim(0,16)
axes[1].grid(True, which="both", linestyle="--", alpha=0.5)

axes[2].plot(df["X"], df["Y3"], marker="^", color="forestgreen", linewidth=2)
if not baseline.empty:
    axes[2].axhline(y=baseline["Y3"].values[0], color="forestgreen", linestyle='--', alpha = 0.7, label = 'Omega = 0')
    axes[2].legend(loc="upper right")
    ticks = sorted(list(axes[2].get_yticks()) + [baseline["Y3"].values[0]])
    axes[2].set_yticks(ticks, labels=[str(int(x)) if float(x).is_integer() else f"{x:.2f}" for x in ticks])
axes[2].set_xlabel("Omega", fontsize=11)
axes[2].set_ylabel("Allignment rate (%)", fontsize=10)
axes[2].set_title("Allignment rate", fontsize=12, fontweight="bold")
axes[2].set_ylim(0,100)
axes[2].set_xlim(0.1/1.4,1000*1.4)
axes[2].grid(True, which="both", linestyle="--", alpha=0.5)

plt.xscale("log")

ticks = [x for x in df["X"] if is125Series(x)]
plt.xticks(ticks, labels=[str(int(x)) if float(x).is_integer() else str(x) for x in ticks])

plt.tight_layout()
plt.show()
