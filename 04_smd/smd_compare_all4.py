#!/usr/bin/env python3
"""
Combined SMD analysis for all 4 GIPR consensus hits.
Loads existing Lurasidone data + newly computed Montelukast/Risperidone/Atovaquone.
Generates:
  - smd_compare_fig1_rupture_bar.png   — bar chart rupture forces ± SD
  - smd_compare_fig2_force_time.png    — 4-panel force vs time
  - smd_compare_fig3_combined_rank.png — final combined ranking table (all metrics)
  - smd_all4_analysis_results.json     — machine-readable summary

Outputs final_top_compound decision to stdout and JSON.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.ndimage import uniform_filter1d

SMD_BASE = Path("/run/media/stalin/Expansion/Sandisk_ssd/FDA/md")
FIG_DIR  = SMD_BASE / "smd_comparison_figures"
FIG_DIR.mkdir(exist_ok=True)

KJ_TO_PN = 1e3 / 6.022e23 / 1e-9 * 1e12   # 1 kJ/mol/nm = 1.661 pN
REP_IDS  = [f"rep{i:02d}" for i in range(1, 6)]

COMPOUNDS = ["Montelukast", "Lurasidone", "Risperidone", "Atovaquone"]
COLORS    = {
    "Montelukast": "#55A868",
    "Lurasidone":  "#C44E52",
    "Risperidone": "#4C72B0",
    "Atovaquone":  "#CCB974",
}

# All-metric data from manuscript (Scons-corrected values)
METRICS = {
    "Montelukast": {"IC50_nM": 8.9,  "dock_kcal": -11.39, "Scons": 0.025, "conf": 0.736, "RotBonds": 12},
    "Lurasidone":  {"IC50_nM": 7.5,  "dock_kcal": -10.01, "Scons": 0.128, "conf": 0.726, "RotBonds": 5},
    "Risperidone": {"IC50_nM": 11.2, "dock_kcal": -9.73,  "Scons": 0.218, "conf": 0.724, "RotBonds": 4},
    "Atovaquone":  {"IC50_nM": 14.1, "dock_kcal": -8.55,  "Scons": 0.378, "conf": 0.688, "RotBonds": 2},
}

# ── Load SMD data ──────────────────────────────────────────────────────────────
def load_smd(compound):
    smd_dir = SMD_BASE / f"smd_{compound}"
    reps = {}
    for rep in REP_IDS:
        fp = smd_dir / rep / "smd_force_extension.json"
        if not fp.exists():
            raise FileNotFoundError(f"Missing: {fp}")
        d   = json.loads(fp.read_text())
        t   = np.array(d["time_ps"])
        ext = np.array(d["extension_nm"]) * 10   # Å
        F   = np.array(d["force_kJ_mol_nm"])
        Fp  = F * KJ_TO_PN
        ri  = np.argmax(F)
        reps[rep] = dict(t_ns=t/1000, ext=ext, F=F, Fp=Fp,
                         Fmax=F[ri], Fmax_pN=Fp[ri],
                         ext_rup=ext[ri], t_rup=t[ri]/1000)
    Fmax_pN = np.array([reps[r]["Fmax_pN"] for r in REP_IDS])
    return reps, Fmax_pN

def smooth(y, w=120):
    return uniform_filter1d(y.astype(float), size=w)

data = {}
for cpd in COMPOUNDS:
    reps, Fmax_pN = load_smd(cpd)
    data[cpd] = {
        "reps":      reps,
        "Fmax_pN":   Fmax_pN,
        "mean_pN":   Fmax_pN.mean(),
        "std_pN":    Fmax_pN.std(ddof=1),
    }
    print(f"  {cpd:14s}: {Fmax_pN.mean():.1f} ± {Fmax_pN.std(ddof=1):.1f} pN  "
          f"(reps: " + "  ".join(f"{v:.0f}" for v in Fmax_pN) + ")")

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.linewidth":    1.2,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Rupture Force Comparison Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#f9f9f9")

x = np.arange(len(COMPOUNDS))
means = [data[c]["mean_pN"] for c in COMPOUNDS]
stds  = [data[c]["std_pN"]  for c in COMPOUNDS]
bar_colors = [COLORS[c] for c in COMPOUNDS]

bars = ax.bar(x, means, yerr=stds, capsize=7, color=bar_colors,
              edgecolor="white", linewidth=0.8, alpha=0.88,
              error_kw={"elinewidth": 2.0, "capthick": 2.0, "ecolor": "#333333"},
              width=0.55, zorder=3)

# Value labels on bars
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, mean + std + 3,
            f"{mean:.0f} ± {std:.0f}", ha="center", fontsize=10.5,
            fontweight="bold", color="#222222")

# Individual rep dots
for xi, cpd in enumerate(COMPOUNDS):
    reps_pN = data[cpd]["Fmax_pN"]
    np.random.seed(42)
    jitter = np.random.uniform(-0.14, 0.14, len(reps_pN))
    ax.scatter(xi + jitter, reps_pN, color="white", s=55, zorder=5,
               edgecolors=COLORS[cpd], linewidths=1.5)

ax.set_xticks(x)
ax.set_xticklabels(COMPOUNDS, fontsize=11.5)
ax.set_ylabel("Max Rupture Force (pN)", fontsize=13)
ax.set_title("SMD Rupture Force Comparison — 4 GIPR Consensus Hits\n"
             "5 replicates × 5 ns per compound  ·  v = 0.5 Å/ns",
             fontsize=12, pad=10)
ax.set_ylim(0, max(means) + max(stds) + 25)
ax.grid(True, axis="y", linestyle=":", alpha=0.4, color="#aaaaaa", zorder=0)
fig.tight_layout()
fig.savefig(FIG_DIR / "smd_compare_fig1_rupture_bar.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("fig1 done")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — 4-panel Force vs Time
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
fig.patch.set_facecolor("white")
rep_colors = ["#C0392B", "#2980B9", "#27AE60", "#D4A017", "#7D3C98"]

for ax, cpd in zip(axes.flat, COMPOUNDS):
    ax.set_facecolor("#f9f9f9")
    reps  = data[cpd]["reps"]
    t_arr = reps["rep01"]["t_ns"]
    Fs    = np.vstack([smooth(reps[r]["Fp"]) for r in REP_IDS])
    mu    = Fs.mean(0)
    sd    = Fs.std(0, ddof=1)

    ax.fill_between(t_arr, mu - sd, mu + sd,
                    color="#BBBBBB", alpha=0.35, zorder=1)

    for i, rep in enumerate(REP_IDS):
        ax.plot(t_arr, smooth(reps[rep]["Fp"]),
                color=rep_colors[i], lw=1.1, alpha=0.65, label=f"Rep {i+1}", zorder=2)
        t_r = reps[rep]["t_rup"]
        F_r = reps[rep]["Fmax_pN"]
        ax.scatter(t_r, F_r, color=rep_colors[i], s=55, zorder=5,
                   edgecolors="white", lw=1.0)

    ax.plot(t_arr, mu, color="black", lw=2.2, zorder=3, label="Mean")

    mean_pN = data[cpd]["mean_pN"]
    std_pN  = data[cpd]["std_pN"]
    ax.axhline(mean_pN, color=COLORS[cpd], lw=1.3, ls="--", alpha=0.8, zorder=2)
    ax.text(4.85, mean_pN + 2,
            f"{mean_pN:.0f} ± {std_pN:.0f} pN",
            ha="right", fontsize=10, color=COLORS[cpd], fontweight="bold")

    ax.set_xlim(0, 5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time (ns)", fontsize=11)
    ax.set_ylabel("Spring Force (pN)", fontsize=11)
    ax.set_title(f"{cpd}", fontsize=12.5, fontweight="bold",
                 color=COLORS[cpd], pad=6)
    ax.legend(fontsize=7.5, ncol=3, loc="upper left",
              framealpha=0.80, edgecolor="none")
    ax.grid(True, linestyle=":", alpha=0.3, color="#aaaaaa", zorder=0)

fig.suptitle("SMD Force vs Time — All 4 GIPR Consensus Hits  (5 reps × 5 ns, v = 0.5 Å/ns)",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "smd_compare_fig2_force_time.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("fig2 done")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Combined Ranking Table (all metrics)
# ══════════════════════════════════════════════════════════════════════════════
# Rank each compound on each metric (1=best, 4=worst)
# Metrics:  Scons (lower=better), IC50 (lower=better), dock (more negative=better),
#           SMD rupture force (higher=better), conf (higher=better)

scons_order = sorted(COMPOUNDS, key=lambda c: METRICS[c]["Scons"])
ic50_order  = sorted(COMPOUNDS, key=lambda c: METRICS[c]["IC50_nM"])
dock_order  = sorted(COMPOUNDS, key=lambda c: METRICS[c]["dock_kcal"])
smd_order   = sorted(COMPOUNDS, key=lambda c: data[c]["mean_pN"], reverse=True)
conf_order  = sorted(COMPOUNDS, key=lambda c: METRICS[c]["conf"], reverse=True)

def rank_of(cpd, ordered_list):
    return ordered_list.index(cpd) + 1

combined_scores = {}
for cpd in COMPOUNDS:
    r_scons = rank_of(cpd, scons_order)
    r_ic50  = rank_of(cpd, ic50_order)
    r_dock  = rank_of(cpd, dock_order)
    r_smd   = rank_of(cpd, smd_order)
    r_conf  = rank_of(cpd, conf_order)
    combined_scores[cpd] = (r_scons + r_ic50 + r_dock + r_smd + r_conf) / 5.0

final_order = sorted(COMPOUNDS, key=lambda c: combined_scores[c])
top_compound = final_order[0]

print(f"\n{'='*60}")
print(f"  FINAL RANKING (mean rank across 5 metrics)")
for rank, cpd in enumerate(final_order, 1):
    print(f"  {rank}. {cpd:14s}  combined score = {combined_scores[cpd]:.2f}")
print(f"\n  *** TOP COMPOUND: {top_compound} ***")
print(f"{'='*60}\n")

# Table figure
fig, ax = plt.subplots(figsize=(12, 4))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.axis("off")

col_labels = ["Compound", "IC₅₀\n(nM)", "Dock\n(kcal/mol)", "Scons",
              "Conf\nscore", "SMD Force\n(pN, mean±SD)",
              "Rank\n(Scons)", "Rank\n(IC₅₀)", "Rank\n(Dock)",
              "Rank\n(SMD)", "Rank\n(Conf)", "Mean\nRank"]

rows = []
for cpd in final_order:
    m   = METRICS[cpd]
    smd = data[cpd]
    row = [
        cpd,
        f"{m['IC50_nM']:.1f}",
        f"{m['dock_kcal']:.2f}",
        f"{m['Scons']:.3f}",
        f"{m['conf']:.3f}",
        f"{smd['mean_pN']:.0f} ± {smd['std_pN']:.0f}",
        str(rank_of(cpd, scons_order)),
        str(rank_of(cpd, ic50_order)),
        str(rank_of(cpd, dock_order)),
        str(rank_of(cpd, smd_order)),
        str(rank_of(cpd, conf_order)),
        f"{combined_scores[cpd]:.2f}",
    ]
    rows.append(row)

tbl = ax.table(
    cellText=rows,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.0, 2.0)

# Style header
for col in range(len(col_labels)):
    tbl[(0, col)].set_facecolor("#D9D9D9")
    tbl[(0, col)].set_text_props(fontweight="bold")

# Highlight top compound
top_row = final_order.index(top_compound) + 1
for col in range(len(col_labels)):
    tbl[(top_row, col)].set_facecolor("#FFF3CD")
    tbl[(top_row, col)].set_text_props(fontweight="bold")

ax.set_title(f"Combined Ranking — All 4 GIPR Consensus Hits\n"
             f"★ Top compound: {top_compound}  (mean rank = {combined_scores[top_compound]:.2f})",
             fontsize=12, fontweight="bold", pad=15)

fig.tight_layout()
fig.savefig(FIG_DIR / "smd_compare_fig3_combined_rank.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("fig3 done")

# ── Save JSON summary ──────────────────────────────────────────────────────────
summary = {
    "top_compound": top_compound,
    "final_ranking": final_order,
    "combined_scores": combined_scores,
    "smd_results": {
        cpd: {
            "mean_pN": float(data[cpd]["mean_pN"]),
            "std_pN":  float(data[cpd]["std_pN"]),
            "per_rep_pN": data[cpd]["Fmax_pN"].tolist(),
        }
        for cpd in COMPOUNDS
    },
    "metric_ranks": {
        cpd: {
            "scons": rank_of(cpd, scons_order),
            "ic50":  rank_of(cpd, ic50_order),
            "dock":  rank_of(cpd, dock_order),
            "smd":   rank_of(cpd, smd_order),
            "conf":  rank_of(cpd, conf_order),
        }
        for cpd in COMPOUNDS
    },
}
out_json = SMD_BASE / "smd_all4_analysis_results.json"
out_json.write_text(json.dumps(summary, indent=2))
print(f"\nAll figures → {FIG_DIR}/")
print(f"JSON summary → {out_json}")
print(f"\n*** FINAL TOP COMPOUND: {top_compound} (mean rank {combined_scores[top_compound]:.2f}) ***")
