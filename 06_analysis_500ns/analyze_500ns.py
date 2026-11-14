#!/usr/bin/env python3
"""
500 ns MD analysis for Montelukast and Lurasidone (top 2 compounds).
Computes: protein RMSD, Rg, ligand RMSD, H-bonds over full 500 ns.
Uses stride=10 to keep memory under 6 GB (5000 frames per compound).
"""
import numpy as np
import mdtraj as md
from pathlib import Path
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE    = Path("/run/media/stalin/Expansion/Sandisk_ssd/FDA/md/production")
OUT_DIR = Path("/run/media/stalin/Expansion/Sandisk_ssd/FDA/md/analysis_500ns")
OUT_DIR.mkdir(exist_ok=True)

COMPOUNDS  = ["Lurasidone", "Montelukast"]
COLORS     = {"Lurasidone": "#C62828", "Montelukast": "#2E7D32"}
TRAJ_STRIDE = 10   # load every 10th frame  →  ~5000 frames per compound
HB_STRIDE   = 5    # additionally subsample for H-bond calc  → every 50th frame

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.linewidth": 1.1, "axes.spines.top": False, "axes.spines.right": False,
})

def smooth(y, w=500):
    if len(y) < w:
        return y.astype(float)
    return uniform_filter1d(y.astype(float), size=w)

data = {}

for cpd in COMPOUNDS:
    print(f"\n{'='*55}\nLoading {cpd} (stride={TRAJ_STRIDE})…\n{'='*55}")
    work   = BASE / cpd
    prmtop = str(work / "complex.prmtop")
    trajs  = sorted(work.glob("traj_*.dcd"))
    print(f"  {len(trajs)} trajectory files: {[t.name for t in trajs]}")

    # Load strided trajectory
    t = md.load([str(f) for f in trajs], top=prmtop, stride=TRAJ_STRIDE)
    print(f"  Loaded: {t.n_frames} frames  ({t.n_frames * 0.01 * TRAJ_STRIDE:.0f} ns sampled)")

    prot_bb = t.topology.select("backbone")
    lig_idx = t.topology.select("resname LIG")
    ca_idx  = t.topology.select("name CA")
    # Time axis: each loaded frame represents TRAJ_STRIDE original frames × 0.01 ns
    time_ns = np.arange(t.n_frames) * TRAJ_STRIDE * 0.01

    ref = t[0]

    # Superpose on protein backbone
    t_sup = t.superpose(ref, atom_indices=prot_bb)

    # Protein backbone RMSD
    rmsd_bb  = md.rmsd(t_sup, ref, atom_indices=prot_bb) * 10   # Å

    # Radius of gyration (Cα only)
    rg = md.compute_rg(t_sup.atom_slice(ca_idx)) * 10            # Å

    # Ligand RMSD (after protein fit)
    rmsd_lig = md.rmsd(t_sup, ref, atom_indices=lig_idx) * 10    # Å

    # H-bonds every HB_STRIDE strided-frames
    t_hb = t_sup[::HB_STRIDE]
    lig_set = set(lig_idx.tolist())
    hb_counts = []
    for fi in range(t_hb.n_frames):
        frame  = t_hb[fi]
        hbonds = md.baker_hubbard(frame, freq=0, periodic=False)
        count  = sum(1 for h in hbonds if h[0] in lig_set or h[2] in lig_set)
        hb_counts.append(count)
    hb_time = np.arange(len(hb_counts)) * HB_STRIDE * TRAJ_STRIDE * 0.01
    hb_arr  = np.array(hb_counts, dtype=float)

    data[cpd] = dict(
        time=time_ns, rmsd_bb=rmsd_bb, rg=rg,
        rmsd_lig=rmsd_lig, hb_time=hb_time, hb=hb_arr
    )

    np.savez(str(OUT_DIR / f"{cpd}_500ns_metrics.npz"),
             time=time_ns, rmsd_bb=rmsd_bb, rg=rg,
             rmsd_lig=rmsd_lig, hb_time=hb_time, hb=hb_arr)

    print(f"  Protein RMSD : {rmsd_bb.mean():.2f} ± {rmsd_bb.std():.2f} Å")
    print(f"  Ligand RMSD  : {rmsd_lig.mean():.2f} ± {rmsd_lig.std():.2f} Å")
    print(f"  Rg (Cα)      : {rg.mean():.2f} ± {rg.std():.2f} Å")
    print(f"  H-bonds      : {hb_arr.mean():.2f} ± {hb_arr.std():.2f}")

    # Free large array to save RAM before next compound
    del t, t_sup, t_hb
    import gc; gc.collect()

# ═══════════════════════════════════════════════════════
# Figure A — 4-panel comparison (Lurasidone vs Montelukast, 500 ns)
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.patch.set_facecolor("white")

metrics_plot = [
    ("rmsd_bb",  "Protein backbone RMSD (Å)"),
    ("rg",       "Radius of gyration, Cα (Å)"),
    ("rmsd_lig", "Ligand RMSD (Å)"),
    (None,       "H-bonds (ligand–protein)"),
]

for ax, (metric, ylabel) in zip(axes.flat, metrics_plot):
    ax.set_facecolor("white")
    for cpd in COMPOUNDS:
        d   = data[cpd]
        col = COLORS[cpd]
        if metric is None:
            y, t_ = d["hb"], d["hb_time"]
        else:
            y, t_ = d[metric], d["time"]

        w_smooth = max(10, min(200, len(y) // 25))
        ax.plot(t_, y, color=col, lw=0.4, alpha=0.25)
        ax.plot(t_, smooth(y, w_smooth), color=col, lw=2.0, label=cpd)
        ax.axhline(y.mean(), color=col, lw=0.9, ls="--", alpha=0.55)

    ax.axvline(100, color="grey", lw=1.2, ls=":", alpha=0.7)
    ymin, ymax = ax.get_ylim()
    ax.text(102, ymax - (ymax - ymin) * 0.05, "100 ns", fontsize=8,
            color="grey", va="top")
    ax.set_xlabel("Time (ns)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, linestyle=":", alpha=0.25, color="#aaaaaa")

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_500ns_md_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nFig A saved → {OUT_DIR}/fig_500ns_md_comparison.png")

# ═══════════════════════════════════════════════════════
# Figure B — Ligand RMSD stability: 0–100 ns vs 400–500 ns
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.patch.set_facecolor("white")

for ax, cpd in zip(axes, COMPOUNDS):
    ax.set_facecolor("white")
    d   = data[cpd]
    t_  = d["time"]
    y   = d["rmsd_lig"]
    col = COLORS[cpd]

    ax.fill_between(t_, 0, y, color=col, alpha=0.10)
    w_sm = max(10, min(200, len(y) // 25))
    ax.plot(t_, smooth(y, w_sm), color=col, lw=2.0)

    m100 = y[t_ <= 100].mean() if (t_ <= 100).any() else np.nan
    m500 = y[t_ >= 400].mean() if (t_ >= 400).any() else np.nan
    ax.axhline(m100, color="navy",       lw=1.3, ls="--",
               label=f"0–100 ns mean: {m100:.2f} Å")
    ax.axhline(m500, color="darkorange", lw=1.3, ls="--",
               label=f"400–500 ns mean: {m500:.2f} Å")

    ax.set_xlim(0, t_.max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time (ns)", fontsize=11)
    ax.set_ylabel("Ligand RMSD (Å)", fontsize=11)
    ax.set_title(cpd, fontsize=13, fontweight="bold", color=col, loc="left")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, linestyle=":", alpha=0.25)

fig.suptitle("Ligand binding-pocket stability: 100 ns vs 500 ns",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_500ns_ligand_rmsd.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Fig B saved → {OUT_DIR}/fig_500ns_ligand_rmsd.png")

# ═══════════════════════════════════════════════════════
print("\n=== DONE ===")
for cpd in COMPOUNDS:
    d = data[cpd]
    print(f"\n{cpd} 500 ns summary:")
    print(f"  Protein RMSD : {d['rmsd_bb'].mean():.2f} ± {d['rmsd_bb'].std():.2f} Å")
    print(f"  Ligand RMSD  : {d['rmsd_lig'].mean():.2f} ± {d['rmsd_lig'].std():.2f} Å")
    print(f"  Rg           : {d['rg'].mean():.2f} ± {d['rg'].std():.2f} Å")
    print(f"  H-bonds      : {d['hb'].mean():.2f} ± {d['hb'].std():.2f}")
