#!/usr/bin/env python
"""SMD Analysis — Lurasidone / GIPR — 3 publication figures"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.ndimage import uniform_filter1d

SMD_DIR = Path("/home/stalin/Downloads/Enamine/GIPR/FDA/md/smd_Lurasidone")
FIG_DIR = SMD_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

REP_IDS = [f"rep{i:02d}" for i in range(1, 6)]
N_REPS  = 5
COLORS  = ["#C0392B", "#2980B9", "#27AE60", "#D4A017", "#7D3C98"]
KJ_TO_PN = 1e3 / 6.022e23 / 1e-9 * 1e12   # 1 kJ/mol/nm = 1.661 pN

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.linewidth":    1.2,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.size":  5,
    "ytick.major.size":  5,
})

def smooth(y, w=120):
    return uniform_filter1d(y.astype(float), size=w)

# ── Load ──────────────────────────────────────────────────────────────────────
reps = {}
for rep in REP_IDS:
    d    = json.loads((SMD_DIR / rep / "smd_force_extension.json").read_text())
    t    = np.array(d["time_ps"])
    ext  = np.array(d["extension_nm"]) * 10   # Å
    F    = np.array(d["force_kJ_mol_nm"])
    Fp   = F * KJ_TO_PN
    ri   = np.argmax(F)
    reps[rep] = dict(t_ns=t/1000, ext=ext, F=F, Fp=Fp,
                     Fmax=F[ri], Fmax_pN=Fp[ri],
                     ext_rup=ext[ri], t_rup=t[ri]/1000)

Fmax_pN  = np.array([reps[r]["Fmax_pN"] for r in REP_IDS])
Fmean_pN = Fmax_pN.mean()
Fstd_pN  = Fmax_pN.std(ddof=1)
t_ns_arr = reps["rep01"]["t_ns"]

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Force vs Time  (all reps + mean ± SD ribbon, dual axis)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

Fs_pN   = np.vstack([smooth(reps[r]["Fp"]) for r in REP_IDS])
mu_pN   = Fs_pN.mean(0)
sd_pN   = Fs_pN.std(0, ddof=1)

ax.fill_between(t_ns_arr, mu_pN - sd_pN, mu_pN + sd_pN,
                color="#BBBBBB", alpha=0.40, zorder=1, label="Mean ± SD")

for i, rep in enumerate(REP_IDS):
    ax.plot(t_ns_arr, smooth(reps[rep]["Fp"]),
            color=COLORS[i], lw=1.3, alpha=0.70, label=f"Rep {i+1}", zorder=2)

ax.plot(t_ns_arr, mu_pN, color="black", lw=2.5, zorder=3, label="Mean")

# Rupture markers — vertical ticks at top
for i, rep in enumerate(REP_IDS):
    t_r, F_r = reps[rep]["t_rup"], reps[rep]["Fmax_pN"]
    ax.axvline(t_r, color=COLORS[i], lw=1.0, ls="--", alpha=0.45, zorder=1)
    ax.scatter(t_r, F_r, color=COLORS[i], s=70, zorder=5,
               edgecolors="white", lw=1.2)

# Mean rupture force band
ax.axhline(Fmean_pN, color="#C0392B", lw=1.3, ls=(0,(6,4)), alpha=0.7, zorder=2)
ax.text(4.85, Fmean_pN + 4,
        f"{Fmean_pN:.0f} ± {Fstd_pN:.0f} pN",
        ha="right", fontsize=10.5, color="#C0392B", fontweight="bold")

ax.set_xlabel("Time (ns)", fontsize=13)
ax.set_ylabel("Spring Force (pN)", fontsize=13)

# Right axis in kJ/mol/nm
ax2 = ax.twinx()
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_linewidth(1.2)
ax2.set_ylim(np.array(ax.get_ylim()) / KJ_TO_PN)
ax2.set_ylabel("Force (kJ mol$^{-1}$ nm$^{-1}$)", fontsize=12)
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

ax.set_ylim(bottom=0)
ax.set_xlim(0, 5)
ax.legend(fontsize=9, ncol=3, loc="upper left",
          framealpha=0.85, edgecolor="none")
ax.set_title("Lurasidone / GIPR — SMD Force vs Time\n"
             "5 replicates × 5 ns,  v = 0.5 Å/ns  ·  dots = rupture peaks",
             fontsize=12, pad=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "smd_fig1_force_time.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("fig1 done")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Rupture Statistics (violin + timing)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 5),
                         gridspec_kw={"width_ratios": [1.1, 1]})

# Left: violin + jitter
ax = axes[0]
vp = ax.violinplot([Fmax_pN], positions=[0], widths=0.55,
                   showmeans=False, showmedians=False, showextrema=False)
vp["bodies"][0].set_facecolor("#2980B9")
vp["bodies"][0].set_alpha(0.28)
vp["bodies"][0].set_edgecolor("#1A5276")
vp["bodies"][0].set_linewidth(1.5)

ax.errorbar(0, Fmean_pN, yerr=Fstd_pN, fmt="D",
            color="black", ms=10, capsize=8, capthick=2,
            elinewidth=2.2, zorder=5)

np.random.seed(7)
jitter = np.random.uniform(-0.09, 0.09, N_REPS)
for i, (j, val) in enumerate(zip(jitter, Fmax_pN)):
    ax.scatter(j, val, color=COLORS[i], s=100, zorder=6,
               edgecolors="white", linewidths=1.3)

# Per-rep labels staggered on right
ymin, ymax = Fmax_pN.min() - 3, Fmax_pN.max() + 3
ax.set_ylim(ymin - 8, ymax + 15)
sorted_idx = np.argsort(Fmax_pN)[::-1]
x_positions = [0.38, 0.38, 0.38, 0.38, 0.38]
for rank, i in enumerate(sorted_idx):
    rep = REP_IDS[i]
    val = reps[rep]["Fmax_pN"]
    y_label = ymin + (ymax - ymin) / (N_REPS - 1) * (N_REPS - 1 - rank) + 2
    ax.annotate(f"Rep {i+1}:  {val:.0f} pN",
                xy=(jitter[i], val),
                xytext=(0.36, y_label),
                fontsize=9.5, color=COLORS[i], fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=COLORS[i],
                                lw=0.9, alpha=0.6))

ax.text(0, ymin - 5,
        f"{Fmean_pN:.0f} ± {Fstd_pN:.0f} pN",
        ha="center", fontsize=11, fontweight="bold", color="black")
ax.set_xlim(-0.45, 0.85)
ax.set_xticks([])
ax.set_ylabel("Max Rupture Force (pN)", fontsize=13)
ax.set_title("Rupture Force Distribution", fontsize=12, pad=8)

# Right: rupture timing
ax = axes[1]
t_rup = np.array([reps[r]["t_rup"] for r in REP_IDS])
t_mean = t_rup.mean();  t_std = t_rup.std(ddof=1)
x = np.arange(1, 6)
bars = ax.bar(x, t_rup, color=COLORS, edgecolor="white", lw=0.8,
              width=0.62, alpha=0.88, zorder=2)
ax.axhline(t_mean, color="#555555", lw=1.5, ls="--", alpha=0.7, zorder=3)
ax.fill_between([0.5, 5.5], t_mean - t_std, t_mean + t_std,
                color="#999999", alpha=0.15, zorder=1)
ax.text(3.0, t_mean + 0.08,
        f"$\\langle t_{{rup}}\\rangle$ = {t_mean:.1f} ± {t_std:.1f} ns",
        ha="center", fontsize=10.5, color="#444444")
for bar, val in zip(bars, t_rup):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.06,
            f"{val:.2f}", ha="center", fontsize=10, color="black")
ax.set_xticks(x)
ax.set_xticklabels([f"Rep {i}" for i in x], fontsize=10.5)
ax.set_ylabel("Time of Rupture (ns)", fontsize=13)
ax.set_title("Rupture Timing", fontsize=12, pad=8)
ax.set_ylim(0, t_rup.max() * 1.35)

fig.suptitle(
    f"Lurasidone / GIPR — Unbinding Statistics  "
    f"($\\langle F_{{\\rm max}}\\rangle$ = {Fmean_pN:.0f} ± {Fstd_pN:.0f} pN)",
    fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "smd_fig2_rupture_stats.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("fig2 done")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Force-Extension (clipped, heavily smoothed, clean)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

# Only plot positive extension (actual pulling region)
ref_ext = reps["rep01"]["ext"]
mask    = ref_ext >= -0.2   # keep from -0.2 Å

Fs_pN_clip = np.vstack([smooth(reps[r]["Fp"], w=150)[mask] for r in REP_IDS])
ext_clip   = ref_ext[mask]
mu_clip    = Fs_pN_clip.mean(0)
sd_clip    = Fs_pN_clip.std(0, ddof=1)

ax.fill_between(ext_clip, mu_clip - sd_clip, mu_clip + sd_clip,
                color="#BBBBBB", alpha=0.40, zorder=1, label="Mean ± SD")

for i, rep in enumerate(REP_IDS):
    F_clip = smooth(reps[rep]["Fp"], w=150)[mask]
    ax.plot(ext_clip, F_clip,
            color=COLORS[i], lw=1.4, alpha=0.72, label=f"Rep {i+1}", zorder=2)

ax.plot(ext_clip, mu_clip, color="black", lw=2.5, zorder=3, label="Mean")

# Mark rupture points on the curve (nearest index on clipped trace)
for i, rep in enumerate(REP_IDS):
    er = reps[rep]["ext_rup"]
    # find nearest point in clipped extension
    idx = np.argmin(np.abs(ext_clip - er))
    F_here = smooth(reps[rep]["Fp"], w=150)[mask][idx]
    ax.scatter(er, F_here, color=COLORS[i], s=80, zorder=5,
               edgecolors="white", lw=1.3, marker="^")

ax.axhline(Fmean_pN, color="#C0392B", lw=1.3, ls=(0,(6,4)), alpha=0.7, zorder=2)
ax.text(ext_clip.max() * 0.98, Fmean_pN + 4,
        f"$\\langle F_{{\\rm max}}\\rangle$ = {Fmean_pN:.0f} ± {Fstd_pN:.0f} pN",
        ha="right", fontsize=10.5, color="#C0392B", fontweight="bold")

# Right axis kJ/mol/nm
ax2 = ax.twinx()
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_linewidth(1.2)
ax2.set_ylim(np.array(ax.get_ylim()) / KJ_TO_PN)
ax2.set_ylabel("Force (kJ mol$^{-1}$ nm$^{-1}$)", fontsize=12)
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

ax.set_ylim(bottom=0)
ax.set_xlim(ext_clip.min(), ext_clip.max())
ax.set_xlabel("Extension (Å)", fontsize=13)
ax.set_ylabel("Spring Force (pN)", fontsize=13)
ax.legend(fontsize=9, ncol=3, loc="upper left",
          framealpha=0.85, edgecolor="none")
ax.set_title("Lurasidone / GIPR — SMD Force-Extension\n"
             "5 replicates × 5 ns,  v = 0.5 Å/ns  ·  ▲ = rupture",
             fontsize=12, pad=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "smd_fig3_force_extension.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("fig3 done")

print(f"\nAll figures → {FIG_DIR}/")
print(f"Rupture force: {Fmean_pN:.0f} ± {Fstd_pN:.0f} pN")
print(f"Per rep (pN) : " + "  ".join(f"{v:.0f}" for v in Fmax_pN))
print(f"Rupture time : " + "  ".join(f"{reps[r]['t_rup']:.2f}" for r in REP_IDS) + " ns")
