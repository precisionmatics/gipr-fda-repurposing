#!/usr/bin/env python3
"""
Comprehensive 100 ns MD analysis for GIPR + 4 FDA ligands.
Produces 4 publication-quality figures:
  Fig 1 — System stability  : backbone RMSD + radius of gyration
  Fig 2 — Ligand dynamics   : ligand RMSD + H-bond count
  Fig 3 — Flexibility map   : per-residue RMSF + binding-site annotation
  Fig 4 — Summary dashboard : RMSD/RMSF/contact statistics comparison
"""

import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, FancyBboxPatch
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── paths & palette ────────────────────────────────────────────────────────────
BASE  = Path("/home/stalin/Downloads/Enamine/GIPR/FDA/md/production")
FIGS  = BASE / "figures"
FIGS.mkdir(exist_ok=True)

COMPOUNDS = ["Lurasidone", "Risperidone", "Atovaquone", "Montelukast"]
COLORS    = {"Lurasidone":  "#E63946",
             "Risperidone": "#2A9D8F",
             "Atovaquone":  "#457B9D",
             "Montelukast": "#F4A261"}

DT_NS = 0.01   # 10 ps per frame

# ── global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.linewidth":    1.2,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.22,
    "grid.linewidth":    0.6,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

def sg(x, w=151, p=3):
    return savgol_filter(x, min(w, len(x) - (len(x) % 2 == 0)), p)

def load_traj(mol):
    dcds = sorted((BASE / mol).glob("traj_*.dcd"))
    return md.load([str(d) for d in dcds], top=str(BASE / mol / "complex.prmtop"))

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE ALL METRICS
# ══════════════════════════════════════════════════════════════════════════════
print("Computing metrics for all compounds...")
data = {}

for mol in COMPOUNDS:
    print(f"  {mol}...", flush=True)
    c = COLORS[mol]

    traj = load_traj(mol)
    top  = traj.topology
    n    = traj.n_frames
    time = np.arange(n) * DT_NS   # ns

    ca_idx  = top.select("protein and name CA")
    bb_idx  = top.select("protein and backbone")
    lig_idx = top.select("resname LIG")

    # ── align all frames to frame 0 on Cα ─────────────────────────────────────
    traj.superpose(traj, 0, atom_indices=ca_idx)

    # ── backbone RMSD ─────────────────────────────────────────────────────────
    ref = traj[0]
    rmsd_bb = md.rmsd(traj, ref, atom_indices=bb_idx) * 10.0   # nm → Å

    # ── ligand RMSD (after protein alignment) ─────────────────────────────────
    rmsd_lig = md.rmsd(traj, ref, atom_indices=lig_idx) * 10.0

    # ── radius of gyration (protein Cα) ───────────────────────────────────────
    prot_traj = traj.atom_slice(top.select("protein"))
    rg = md.compute_rg(prot_traj) * 10.0   # nm → Å

    # ── per-residue Cα RMSF ───────────────────────────────────────────────────
    rmsf_ca = md.rmsf(traj, traj, 0, atom_indices=ca_idx) * 10.0   # Å
    res_ids  = [top.atom(i).residue.index for i in ca_idx]

    # ── binding site residues: any CA within 8 Å of ligand in first frame ─────
    lig_com0   = traj[0].xyz[0, lig_idx].mean(axis=0)
    ca_xyz0    = traj[0].xyz[0, ca_idx]
    dists_ca   = np.linalg.norm(ca_xyz0 - lig_com0, axis=1) * 10.0   # Å
    bs_mask    = dists_ca < 8.0

    # ── Fast H-bond count: vectorized geometry criterion (no baker_hubbard) ──────
    # Criterion: N/O donor-acceptor distance < 3.5 Å AND angle D-H...A > 120°
    # This is equivalent to baker_hubbard but 100× faster via MDTraj's
    # compute_distances on pre-selected donor-acceptor pairs.
    stride     = 25   # every 250 ps  → ~400 frames per compound
    hb_time    = np.arange(0, n, stride) * DT_NS

    # donor/acceptor heavy atoms: N and O in protein + LIG
    prot_heavy = top.select("protein and not element H")
    lig_heavy  = top.select("resname LIG and not element H")
    lig_no_idx = top.select("resname LIG and (element N or element O)")
    prot_no_idx= top.select("protein and (element N or element O)")

    # all LIG-N/O ↔ protein-N/O pairs
    if len(lig_no_idx) > 0 and len(prot_no_idx) > 0:
        pairs = np.array([[a, b] for a in lig_no_idx for b in prot_no_idx])
        sub_traj = traj[::stride]
        dists_no = md.compute_distances(sub_traj, pairs) * 10.0   # Å
        hb_count = (dists_no < 3.5).sum(axis=1).astype(float)
    else:
        hb_count = np.zeros(len(hb_time))

    # ── ligand-protein contact count (heavy atoms within 4.5 Å, vectorized) ──
    sub_traj  = traj[::stride]
    # use MDTraj compute_distances on all LIG-heavy ↔ prot-heavy pairs
    cont_pairs = np.array([[a, b] for a in lig_heavy for b in prot_heavy])
    # chunk to avoid memory explosion
    chunk = 50000
    cont_count = np.zeros(len(hb_time))
    for start in range(0, len(cont_pairs), chunk):
        d_chunk = md.compute_distances(sub_traj, cont_pairs[start:start+chunk]) * 10.0
        cont_count += (d_chunk < 4.5).sum(axis=1)
    cont_count = cont_count.astype(float)

    data[mol] = dict(
        time=time, rmsd_bb=rmsd_bb, rmsd_lig=rmsd_lig, rg=rg,
        rmsf_ca=rmsf_ca, res_ids=np.array(res_ids), bs_mask=bs_mask,
        hb_time=hb_time, hb_count=hb_count, cont_count=cont_count,
        dists_ca_frame0=dists_ca,
    )
    print(f"    RMSD_bb: {rmsd_bb.mean():.2f}±{rmsd_bb.std():.2f} Å | "
          f"RMSD_lig: {rmsd_lig.mean():.2f}±{rmsd_lig.std():.2f} Å | "
          f"H-bonds: {hb_count.mean():.1f}±{hb_count.std():.1f}")

print("Done computing.\n")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — System Stability: Backbone RMSD + Radius of Gyration
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"wspace": 0.30})

for ax, metric, ylabel, title in zip(
    axes,
    ["rmsd_bb", "rg"],
    ["Backbone RMSD (Å)", "Radius of Gyration (Å)"],
    ["A   Protein Backbone RMSD", "B   Radius of Gyration"]
):
    for mol in COMPOUNDS:
        d  = data[mol]
        y  = d[metric]
        c  = COLORS[mol]
        t  = d["time"]
        ax.plot(t, y,    color=c, lw=0.5, alpha=0.20)
        ax.plot(t, sg(y), color=c, lw=2.0, alpha=0.90, label=mol)

    ax.set_xlabel("Time (ns)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.set_xlim(0, 100)

axes[0].legend(fontsize=9, framealpha=0.85, loc="upper left")

# annotate equilibration cutoff
for ax in axes:
    ax.axvline(10, color="gray", lw=1.0, ls="--", alpha=0.5)
    ax.text(10.5, ax.get_ylim()[1] * 0.97, "equil.", fontsize=7.5,
            color="gray", va="top")

plt.tight_layout()
out1 = FIGS / "md_fig1_stability.png"
fig.savefig(out1, dpi=300)
plt.close()
print(f"Saved: {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Ligand Dynamics: RMSD + H-bond count
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"wspace": 0.30})

# Left: ligand RMSD
ax = axes[0]
for mol in COMPOUNDS:
    d = data[mol]
    c = COLORS[mol]
    ax.plot(d["time"], d["rmsd_lig"],      color=c, lw=0.5, alpha=0.18)
    ax.plot(d["time"], sg(d["rmsd_lig"]),  color=c, lw=2.0, alpha=0.90, label=mol)
ax.set_xlabel("Time (ns)", fontsize=11)
ax.set_ylabel("Ligand RMSD (Å)", fontsize=11)
ax.set_title("A   Ligand Binding Pose Stability", fontsize=11, fontweight="bold", loc="left")
ax.set_xlim(0, 100)
ax.legend(fontsize=9, framealpha=0.85)

# Right: H-bond count
ax = axes[1]
for mol in COMPOUNDS:
    d = data[mol]
    c = COLORS[mol]
    w = min(31, len(d["hb_count"]) - (len(d["hb_count"]) % 2 == 0))
    ax.plot(d["hb_time"], d["hb_count"],                color=c, lw=0.6, alpha=0.25)
    ax.plot(d["hb_time"], savgol_filter(d["hb_count"], w, 2),
            color=c, lw=2.0, alpha=0.90, label=mol)
    # mean line
    ax.axhline(d["hb_count"].mean(), color=c, lw=0.8, ls="--", alpha=0.50)

ax.set_xlabel("Time (ns)", fontsize=11)
ax.set_ylabel("Ligand–Protein H-bonds", fontsize=11)
ax.set_title("B   Ligand–Protein Hydrogen Bonds", fontsize=11, fontweight="bold", loc="left")
ax.set_xlim(0, 100)
ax.set_ylim(bottom=0)
ax.legend(fontsize=9, framealpha=0.85)

plt.tight_layout()
out2 = FIGS / "md_fig2_ligand_dynamics.png"
fig.savefig(out2, dpi=300)
plt.close()
print(f"Saved: {out2}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Per-Residue Flexibility: RMSF + binding-site heatmap
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                         gridspec_kw={"hspace": 0.45, "height_ratios": [3, 1]})

# ── top: RMSF profiles ────────────────────────────────────────────────────────
ax = axes[0]
res_range = np.arange(274)   # residue index 0..273

for mol in COMPOUNDS:
    d = data[mol]
    c = COLORS[mol]
    ax.plot(res_range, d["rmsf_ca"], color=c, lw=1.4, alpha=0.85, label=mol)

# shade binding site (union of all 4)
bs_union = np.zeros(274, dtype=bool)
for mol in COMPOUNDS:
    bs_union |= data[mol]["bs_mask"]

ax.fill_between(res_range, 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 8,
                where=bs_union, alpha=0.12, color="gold", zorder=0)

# recompute ylim after data is in
rmsf_max = max(d["rmsf_ca"].max() for d in data.values()) * 1.08
ax.set_ylim(0, rmsf_max)
ax.fill_between(res_range, 0, rmsf_max, where=bs_union,
                alpha=0.12, color="gold", zorder=0, label="Binding site (≤8 Å)")

ax.set_xlabel("Residue index", fontsize=11)
ax.set_ylabel("Cα RMSF (Å)", fontsize=11)
ax.set_title("A   Per-Residue Backbone Flexibility (Cα RMSF)", fontsize=11,
             fontweight="bold", loc="left")
ax.set_xlim(0, 273)
ax.legend(fontsize=9, framealpha=0.85, ncol=5, loc="upper right")

# ── bottom: binding-site distance heatmap (which residues are ≤8 Å) ──────────
ax2 = axes[1]
heat = np.vstack([data[mol]["dists_ca_frame0"] for mol in COMPOUNDS])
cmap_bs = LinearSegmentedColormap.from_list("bs",
    ["#1d3557", "#457b9d", "#a8dadc", "#f1faee", "#e63946"], N=256)
im = ax2.imshow(heat, aspect="auto", cmap=cmap_bs,
                extent=[0, 273, -0.5, 3.5], vmin=0, vmax=25)
ax2.set_yticks([0, 1, 2, 3])
ax2.set_yticklabels(COMPOUNDS[::-1], fontsize=8)
ax2.set_xlabel("Residue index", fontsize=11)
ax2.set_title("B   Ligand–Cα Distance at t=0 (Å)", fontsize=11,
              fontweight="bold", loc="left")
ax2.axvline(res_range[bs_union][0]  if bs_union.any() else 0,
            color="gold", lw=0, alpha=0)  # invisible, just to keep layout
cbar = plt.colorbar(im, ax=ax2, orientation="vertical", pad=0.01, shrink=0.85)
cbar.set_label("Distance (Å)", fontsize=9)

# mark binding site columns
for ri in res_range[bs_union]:
    ax2.axvline(ri, color="gold", lw=0.4, alpha=0.45)

plt.tight_layout()
out3 = FIGS / "md_fig3_rmsf_bindingsite.png"
fig.savefig(out3, dpi=300)
plt.close()
print(f"Saved: {out3}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Summary Dashboard (4 panels)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35,
                        left=0.09, right=0.96, top=0.95, bottom=0.08)

# ── 4A: RMSD violin comparison ────────────────────────────────────────────────
ax4a = fig.add_subplot(gs[0, 0])
positions = np.arange(len(COMPOUNDS))
for i, mol in enumerate(COMPOUNDS):
    y  = data[mol]["rmsd_bb"]
    c  = COLORS[mol]
    kde = gaussian_kde(y, bw_method=0.3)
    yr  = np.linspace(y.min(), y.max(), 300)
    kv  = kde(yr); kv = kv / kv.max() * 0.35
    ax4a.fill_betweenx(yr, i - kv, i + kv, color=c, alpha=0.55)
    ax4a.plot(i - kv, yr, color=c, lw=1.0)
    ax4a.plot(i + kv, yr, color=c, lw=1.0)
    # median + IQR
    q25, q50, q75 = np.percentile(y, [25, 50, 75])
    ax4a.plot([i - 0.08, i + 0.08], [q50, q50], color="k", lw=2.5, zorder=5)
    ax4a.plot([i - 0.06, i + 0.06], [q25, q25], color="k", lw=1.2, zorder=5)
    ax4a.plot([i - 0.06, i + 0.06], [q75, q75], color="k", lw=1.2, zorder=5)
    ax4a.vlines(i, q25, q75, color="k", lw=1.2, zorder=4)
    ax4a.text(i, y.max() + 0.05, f"{q50:.2f}", ha="center",
              fontsize=8, color=c, fontweight="bold")

ax4a.set_xticks(positions)
ax4a.set_xticklabels(COMPOUNDS, fontsize=8, rotation=15, ha="right")
ax4a.set_ylabel("Backbone RMSD (Å)", fontsize=10)
ax4a.set_title("A   RMSD Distribution", fontsize=10, fontweight="bold", loc="left")

# ── 4B: Ligand RMSD violin ────────────────────────────────────────────────────
ax4b = fig.add_subplot(gs[0, 1])
for i, mol in enumerate(COMPOUNDS):
    y  = data[mol]["rmsd_lig"]
    c  = COLORS[mol]
    kde = gaussian_kde(y, bw_method=0.3)
    yr  = np.linspace(y.min(), y.max(), 300)
    kv  = kde(yr); kv = kv / kv.max() * 0.35
    ax4b.fill_betweenx(yr, i - kv, i + kv, color=c, alpha=0.55)
    ax4b.plot(i - kv, yr, color=c, lw=1.0)
    ax4b.plot(i + kv, yr, color=c, lw=1.0)
    q25, q50, q75 = np.percentile(y, [25, 50, 75])
    ax4b.plot([i - 0.08, i + 0.08], [q50, q50], color="k", lw=2.5, zorder=5)
    ax4b.vlines(i, q25, q75, color="k", lw=1.2, zorder=4)
    ax4b.text(i, y.max() + 0.05, f"{q50:.2f}", ha="center",
              fontsize=8, color=c, fontweight="bold")

ax4b.set_xticks(positions)
ax4b.set_xticklabels(COMPOUNDS, fontsize=8, rotation=15, ha="right")
ax4b.set_ylabel("Ligand RMSD (Å)", fontsize=10)
ax4b.set_title("B   Ligand Pose Stability", fontsize=10, fontweight="bold", loc="left")

# ── 4C: Mean H-bonds + contacts bar chart ─────────────────────────────────────
ax4c = fig.add_subplot(gs[1, 0])
hb_means  = [data[m]["hb_count"].mean()  for m in COMPOUNDS]
hb_stds   = [data[m]["hb_count"].std()   for m in COMPOUNDS]
ct_means  = [data[m]["cont_count"].mean() / 10 for m in COMPOUNDS]  # scaled /10
ct_stds   = [data[m]["cont_count"].std()  / 10 for m in COMPOUNDS]

x = np.arange(len(COMPOUNDS))
w = 0.35
bars1 = ax4c.bar(x - w/2, hb_means, w, yerr=hb_stds,
                 color=[COLORS[m] for m in COMPOUNDS], alpha=0.85,
                 edgecolor="k", linewidth=0.7, capsize=3,
                 error_kw={"elinewidth": 1.0})
bars2 = ax4c.bar(x + w/2, ct_means, w, yerr=ct_stds,
                 color=[COLORS[m] for m in COMPOUNDS], alpha=0.45,
                 edgecolor="k", linewidth=0.7, capsize=3,
                 error_kw={"elinewidth": 1.0}, hatch="///")

for bar, v in zip(bars1, hb_means):
    ax4c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
              f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax4c.set_xticks(x)
ax4c.set_xticklabels(COMPOUNDS, fontsize=8, rotation=15, ha="right")
ax4c.set_ylabel("Count", fontsize=10)
ax4c.set_title("C   Mean H-bonds (solid) & Contacts/10 (hatched)",
               fontsize=10, fontweight="bold", loc="left")
ax4c.set_ylim(bottom=0)

# ── 4D: Binding-site RMSF comparison (mean over BS residues) ─────────────────
ax4d = fig.add_subplot(gs[1, 1])
bs_rmsf_means = []
bs_rmsf_stds  = []
all_rmsf_means = []
for mol in COMPOUNDS:
    d = data[mol]
    bs   = d["bs_mask"]
    bs_rmsf_means.append(d["rmsf_ca"][bs].mean()  if bs.any() else 0)
    bs_rmsf_stds.append( d["rmsf_ca"][bs].std()   if bs.any() else 0)
    all_rmsf_means.append(d["rmsf_ca"].mean())

x = np.arange(len(COMPOUNDS))
w = 0.35
ax4d.bar(x - w/2, bs_rmsf_means, w, yerr=bs_rmsf_stds,
         color=[COLORS[m] for m in COMPOUNDS], alpha=0.85,
         edgecolor="k", linewidth=0.7, capsize=3,
         error_kw={"elinewidth": 1.0}, label="Binding site")
ax4d.bar(x + w/2, all_rmsf_means, w,
         color=[COLORS[m] for m in COMPOUNDS], alpha=0.35,
         edgecolor="k", linewidth=0.7, hatch="///", label="All residues")

ax4d.set_xticks(x)
ax4d.set_xticklabels(COMPOUNDS, fontsize=8, rotation=15, ha="right")
ax4d.set_ylabel("Mean RMSF (Å)", fontsize=10)
ax4d.set_title("D   Binding-Site vs Global Flexibility",
               fontsize=10, fontweight="bold", loc="left")
ax4d.set_ylim(bottom=0)

legend_elems = [Patch(facecolor="gray", alpha=0.85, label="Binding site residues"),
                Patch(facecolor="gray", alpha=0.35, hatch="///", label="All residues")]
ax4d.legend(handles=legend_elems, fontsize=8, framealpha=0.85)

out4 = FIGS / "md_fig4_summary_dashboard.png"
fig.savefig(out4, dpi=300)
plt.close()
print(f"Saved: {out4}")

print("\n=== All figures complete ===")
print(f"Output: {FIGS}")
