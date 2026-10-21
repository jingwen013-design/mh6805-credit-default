"""
make_context_figures.py — context figures for the slides (not part of the modelling pipeline).   Owner: shared.

Reads only the two official series saved under data/external/ (each CSV starts with comment lines giving the
source URL, access date and licence) and writes figures/context/*.png. Nothing is fitted; the UCI data are not read.

  fsc_taiwan_card_revolving_2004_2009.csv  Financial Supervisory Commission (Taiwan), Banking Bureau, credit-card
                                           time series (index-7.xlsx), revolving balance, NT$ million, end of month.
                                           Open Government Data Licence (Taiwan) v1.0, reuse with attribution.
  frb_us_card_delinquency_1991_2026.csv    Board of Governors of the Federal Reserve System, Charge-Off and
                                           Delinquency Rates, credit cards, all banks, SA (FRED DRCCLACBS). Public domain.

Sizes: the Taiwan chart is drawn for a ~706 px slot on a 1920 px slide (text >= 15 px on the slide); the US chart for
a ~860 px slot. Run from Project/code:  python3 make_context_figures.py
"""
import csv
import datetime as dt
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXT = os.path.join(ROOT, "data", "external")
OUT = os.path.join(ROOT, "figures", "context")
os.makedirs(OUT, exist_ok=True)
NAVY, BLUE, INK2, INK3, GRID = "#18395b", "#2a78d6", "#3d4b57", "#5e7182", "#e3e6ea"


def rows(name):
    with open(os.path.join(EXT, name), encoding="utf-8") as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9aa5b1")
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2)


# ---------------------------------------------------------------- 1. Taiwan revolving balances, slide 2
r = rows("fsc_taiwan_card_revolving_2004_2009.csv")
d = [dt.date(int(x["year"]), int(x["month"]), 15) for x in r]
v = [int(x["revolving_balance_ntd_mn"]) / 1000.0 for x in r]          # NT$ billion
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14})
fig, ax = plt.subplots(figsize=(8.0, 3.68), dpi=240)                     # 1920 x 883 px
ax.axvspan(dt.date(2005, 4, 1), dt.date(2005, 10, 1), color=BLUE, alpha=0.12, lw=0, zorder=0)
ax.axvline(dt.date(2005, 10, 15), color=BLUE, ls=(0, (1.5, 2.2)), lw=2.2, zorder=1)
ax.plot(d, v, color=NAVY, lw=2.6, zorder=3)
ip = max(range(len(v)), key=v.__getitem__)
i08 = d.index(dt.date(2008, 12, 15))
assert d[ip] == dt.date(2005, 11, 15), d[ip]
ax.plot([d[ip]], [v[ip]], "o", color=NAVY, ms=7, zorder=4)
ax.plot([d[i08]], [v[i08]], "o", color=NAVY, ms=7, zorder=4)
drop = 100 * (v[i08] / v[ip] - 1)
ax.annotate(f"peak NT${v[ip]:.0f} bn, Nov 2005", (d[ip], v[ip]), xytext=(dt.date(2006, 3, 1), 560),
            fontsize=14, color=INK2, arrowprops=dict(arrowstyle="-", color=INK3, lw=1))
ax.annotate(f"NT${v[i08]:.0f} bn, Dec 2008 (−{abs(drop):.0f}%)", (d[i08], v[i08]), xytext=(dt.date(2007, 5, 1), 120),
            fontsize=14, color=INK2, arrowprops=dict(arrowstyle="-", color=INK3, lw=1))
ax.text(dt.date(2005, 3, 20), 120, "UCI features\nApr–Sep 2005", ha="right", va="bottom", fontsize=13, color=BLUE)
ax.text(dt.date(2005, 11, 5), 40, "label: Oct 2005", ha="left", va="bottom", fontsize=13, color=BLUE)
ax.set_ylim(0, 620)
ax.set_xlim(dt.date(2004, 1, 1), dt.date(2009, 12, 31))
ax.set_ylabel("NT$ billion", color=INK2)
ax.set_title("Taiwan: credit-card revolving balances, end of month", loc="left", fontsize=15, color=NAVY, fontweight="bold")
style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ctx_taiwan_card_revolving.png"), facecolor="white")
plt.close(fig)
print(f"wrote ctx_taiwan_card_revolving.png  peak {v[ip]:.1f} bn {d[ip]:%b %Y}; Dec 2008 {v[i08]:.1f} bn ({drop:.1f}%)")

# ---------------------------------------------------------------- 2. US card delinquency, appendix A2
r = rows("frb_us_card_delinquency_1991_2026.csv")
q = [int(x["year"]) + (int(x["quarter"]) - 0.5) / 4 for x in r]
p = [float(x["credit_card_delinquency_rate_pct_sa"]) for x in r]
fig, ax = plt.subplots(figsize=(8.0, 3.68), dpi=240)
ax.axvspan(2005.25, 2005.75, color=BLUE, alpha=0.15, lw=0)
ax.plot(q, p, color=NAVY, lw=2.4)
ih = max(range(len(p)), key=p.__getitem__)
il = min(range(len(p)), key=p.__getitem__)
for i, txt, xy in [(ih, f"{p[ih]:.2f}% ({r[ih]['period']})", (q[ih] + 1.0, p[ih] - 0.2)),
                   (il, f"{p[il]:.2f}% ({r[il]['period']})", (q[il] - 9.5, p[il] - 0.9)),
                   (len(p) - 1, f"{p[-1]:.2f}% ({r[-1]['period']})", (q[-1] - 6.5, p[-1] + 1.1))]:
    ax.plot([q[i]], [p[i]], "o", color=NAVY, ms=6)
    ax.annotate(txt, (q[i], p[i]), xytext=xy, fontsize=13, color=INK2, arrowprops=dict(arrowstyle="-", color=INK3, lw=1))
ax.text(2005.0, 0.35, "UCI window\n(Taiwan, 2005)", ha="right", fontsize=12, color=BLUE)
ax.set_ylim(0, 7.6)
ax.set_xlim(1990.5, 2027)
ax.set_ylabel("% of card loans", color=INK2)
ax.set_title("US credit-card delinquency rate, quarterly, 1991–2026", loc="left", fontsize=15, color=NAVY, fontweight="bold")
style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ctx_us_card_delinquency.png"), facecolor="white")
plt.close(fig)
print(f"wrote ctx_us_card_delinquency.png  max {p[ih]} {r[ih]['period']}; min {p[il]} {r[il]['period']}; last {p[-1]} {r[-1]['period']}")
