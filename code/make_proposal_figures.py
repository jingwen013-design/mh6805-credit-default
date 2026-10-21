"""Two compact composite figures for the proposal. Descriptive statistics use the TRAINING partition only
(20,975 clients); panel 1(a) shows the class balance of the full cleaned sample because it describes the target."""
import numpy as np, pandas as pd, matplotlib, os
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load_raw, clean, load_split, FIGURES, TARGET
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12.5, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5})
full = clean(load_raw())
X_tr, X_te, y_tr, y_te, _ = load_split(); df = X_tr.copy(); df[TARGET] = y_tr.values
# ---------------- Figure 1
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5), gridspec_kw={"width_ratios": [0.85, 1.55, 1.15]})
cnt = full[TARGET].value_counts().sort_index()
bars = ax[0].bar(["No default (0)", "Default (1)"], cnt.values, color=["#4C72B0", "#C44E52"])
for b, v in zip(bars, cnt.values): ax[0].text(b.get_x() + b.get_width()/2, v + 300, f"{v:,}\n({v/len(full):.1%})", ha="center", fontsize=11)
ax[0].set_ylim(0, cnt.max() * 1.28); ax[0].set_title("(a) Target (all 29,965 clients)"); ax[0].set_ylabel("clients")
g = df.groupby("PAY_1")[TARGET].agg(["mean", "size"]); g = g[g["size"] >= 20]
labels = {-2: "−2\nno use", -1: "−1\npaid", 0: "0\nrevolv.", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
x = [labels.get(k, str(k)) for k in g.index]
b2 = ax[1].bar(x, g["mean"], color=["#4C72B0" if k <= 0 else "#C44E52" for k in g.index])
for b, (m, n) in zip(b2, g[["mean", "size"]].values): ax[1].text(b.get_x() + b.get_width()/2, m + 0.015, f"{m:.0%}\nn={int(n):,}", ha="center", fontsize=9.5)
ax[1].set_ylim(0, 1.0); ax[1].set_title("(b) Default rate by PAY_1 (training set)"); ax[1].set_xlabel("repayment status in September 2005 (months of delay)"); ax[1].set_ylabel("default rate")
q = pd.qcut(df["LIMIT_BAL"], 5, duplicates="drop"); g3 = df.groupby(q, observed=True)[TARGET].mean()
lab3 = [f"{round(iv.left/1000)}–{round(iv.right/1000)}k" for iv in g3.index]
b3 = ax[2].bar(lab3, g3.values, color="#55A868")
for b, m in zip(b3, g3.values): ax[2].text(b.get_x() + b.get_width()/2, m + 0.006, f"{m:.1%}", ha="center", fontsize=11)
ax[2].set_ylim(0, 0.4); ax[2].set_title("(c) Default rate by credit limit"); ax[2].set_xlabel("LIMIT_BAL quintile (NT$)"); ax[2].set_ylabel("default rate"); ax[2].tick_params(axis="x", labelsize=10.5, rotation=15)
fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "proposal_fig1.png"), dpi=200); plt.close(fig)
# ---------------- Figure 2
fig, ax = plt.subplots(1, 4, figsize=(12.5, 3.3), gridspec_kw={"width_ratios": [1, 1, 1.1, 1.45]})
age = pd.cut(df["AGE"], [20, 25, 30, 40, 50, 60, 80], labels=["21–25", "26–30", "31–40", "41–50", "51–60", "61+"])
ga = df.groupby(age, observed=True)[TARGET].mean(); ax[0].bar(ga.index.astype(str), ga.values, color="#8172B2")
for i, m in enumerate(ga.values): ax[0].text(i, m + 0.005, f"{m:.0%}", ha="center", fontsize=10.5)
ax[0].set_ylim(0, 0.35); ax[0].set_title("(a) by age"); ax[0].tick_params(axis="x", labelsize=9.5, rotation=45)
edu = df["EDUCATION"].map({1: "grad", 2: "univ", 3: "high\nschool", 4: "other"}); ge = df.groupby(edu)[TARGET].mean().reindex(["grad", "univ", "high\nschool", "other"])
ax[1].bar(ge.index, ge.values, color="#CCB974")
for i, m in enumerate(ge.values): ax[1].text(i, m + 0.005, f"{m:.0%}", ha="center", fontsize=10.5)
ax[1].set_ylim(0, 0.35); ax[1].set_title("(b) by education"); ax[1].tick_params(axis="x", labelsize=10.5)
gn = df.groupby("N_MONTHS_DELAYED")[TARGET].mean(); ax[2].bar(gn.index.astype(str), gn.values, color="#DD8452")
for i, m in enumerate(gn.values): ax[2].text(i, m + 0.01, f"{m:.0%}", ha="center", fontsize=10.5)
ax[2].set_ylim(0, 1.0); ax[2].set_title("(c) by months in arrears"); ax[2].set_xlabel("N_MONTHS_DELAYED (engineered)")
num = df.drop(columns=["SEX", "EDUCATION", "MARRIAGE"]); corr = num.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).head(12)[::-1]
ax[3].barh(corr.index, corr.values, color=["#C44E52" if v > 0 else "#4C72B0" for v in corr.values]); ax[3].axvline(0, color="k", lw=0.8)
ax[3].set_title("(d) Pearson r with target"); ax[3].tick_params(axis="y", labelsize=10); ax[3].set_xlim(-0.2, 0.45)
fig.tight_layout(rect=(0, 0, 0.995, 1)); fig.savefig(os.path.join(FIGURES, "proposal_fig2.png"), dpi=200); plt.close(fig)
print("saved (training partition):", {k: round(v, 4) for k, v in g["mean"].to_dict().items()}, "| limit quintiles", [round(v, 4) for v in g3.values], "| zero PAY_AMT1 share", round((df.PAY_AMT1 == 0).mean(), 4), "| SEX", df.groupby("SEX")[TARGET].mean().round(4).to_dict(), "| neg bill any month", int((df[[f'BILL_AMT{i}' for i in range(1,7)]] < 0).any(axis=1).sum()), "| max |r|", round(corr.abs().max(), 3))
