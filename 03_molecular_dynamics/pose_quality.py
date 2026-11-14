#!/usr/bin/env python3
"""
Pose quality analysis for Lurasidone-GIPR MD trajectory.
Computes: RMSD, RMSF, key contacts, per-residue interaction energies.
"""
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms, contacts
import warnings
warnings.filterwarnings('ignore')

WORK = "/home/stalin/Downloads/Enamine/GIPR/FDA/md/production/Lurasidone"
TOP  = f"{WORK}/complex.prmtop"
TRAJ = [f"{WORK}/traj_001.dcd", f"{WORK}/traj_002.dcd"]

print("Loading trajectory...")
u = mda.Universe(TOP, TRAJ)
n_frames = len(u.trajectory)
print(f"  {n_frames} frames, {u.atoms.n_atoms} atoms")

lig   = u.select_atoms("resname LIG")
prot  = u.select_atoms("protein")
print(f"  Ligand: {lig.n_atoms} atoms")
print(f"  Protein: {prot.n_atoms} atoms in {len(prot.residues)} residues")

# ── 1. LIGAND RMSD ────────────────────────────────────────────────────────────
print("\n── 1. Ligand RMSD over 100 ns ──────────────────────────────────────────")
u.trajectory[0]
ref_pos = lig.positions.copy()

rmsd_vals = []
times = []
for ts in u.trajectory:
    mob = lig.positions - lig.center_of_mass()
    ref = ref_pos - ref_pos.mean(axis=0)
    # simple RMSD without superposition (ligand COM-centered)
    rmsd = np.sqrt(((mob - ref)**2).mean())
    rmsd_vals.append(rmsd)
    times.append(ts.time / 1000)  # ns

rmsd_arr = np.array(rmsd_vals)
t_arr    = np.array(times)

print(f"  Mean RMSD: {rmsd_arr.mean():.2f} Å")
print(f"  Max  RMSD: {rmsd_arr.max():.2f} Å  at {t_arr[rmsd_arr.argmax()]:.1f} ns")
print(f"  Last 50ns mean: {rmsd_arr[n_frames//2:].mean():.2f} Å")

# Classify stability
if rmsd_arr.mean() < 2.0:
    stability = "STABLE (< 2 Å average)"
elif rmsd_arr.mean() < 4.0:
    stability = "MODERATE (2-4 Å average)"
else:
    stability = "UNSTABLE (> 4 Å average — ligand likely left pocket)"
print(f"  Stability: {stability}")

# ── 2. KEY CONTACTS (< 4 Å) ──────────────────────────────────────────────────
print("\n── 2. Key protein-ligand contacts ──────────────────────────────────────")
# Sample every 100th frame for speed
contact_count = {}
n_sampled = 0
for ts in u.trajectory[::100]:
    for res in prot.residues:
        res_atoms = res.atoms
        dists = np.linalg.norm(
            res_atoms.positions[:, None, :] - lig.positions[None, :, :], axis=2
        )
        if dists.min() < 4.0:
            key = (res.resid, res.resname)
            contact_count[key] = contact_count.get(key, 0) + 1
    n_sampled += 1

# Sort by frequency
sorted_contacts = sorted(contact_count.items(), key=lambda x: -x[1])
print(f"  Contacts sampled over {n_sampled} frames (every 100th):")
print(f"  {'Residue':<12} {'Resname':<8} {'Frequency':>10} {'%':>6}")
print(f"  {'-'*40}")
for (resid, resname), cnt in sorted_contacts[:20]:
    pct = cnt / n_sampled * 100
    bar = '#' * int(pct / 5)
    print(f"  {resid:<12} {resname:<8} {cnt:>10} {pct:>5.1f}%  {bar}")

# ── 3. INTERACTION ENERGY DECOMPOSITION ──────────────────────────────────────
print("\n── 3. Per-residue VDW + Elec interaction (last 10 ns, sampled) ─────────")
print("  [Using pairwise distance-based energy proxy]")

# Use last 10 ns frames (every 50th)
last_start = max(0, n_frames - n_frames // 10)
lig_pos_frames = []
for ts in u.trajectory[last_start::50]:
    lig_pos_frames.append(lig.positions.copy())

if not lig_pos_frames:
    print("  (no frames available)")
else:
    lig_pos_arr = np.array(lig_pos_frames)  # (n_frames, n_lig, 3)
    mean_lig = lig_pos_arr.mean(axis=0)     # mean ligand position

    # for each residue in contact, count close contacts
    print(f"  {'Resid':<8} {'Resname':<8} {'<3.5A contacts':<16} {'<5A contacts'}")
    print(f"  {'-'*50}")
    for (resid, resname), cnt in sorted_contacts[:15]:
        if cnt / n_sampled < 0.5:  # only persistently-present contacts
            continue
        res_atoms = u.select_atoms(f"protein and resid {resid}")
        n_close = 0
        n_medium = 0
        for lig_frame_pos in lig_pos_arr:
            dists = np.linalg.norm(
                res_atoms.positions[:, None, :] - lig_frame_pos[None, :, :], axis=2
            )
            n_close  += (dists < 3.5).sum()
            n_medium += (dists < 5.0).sum()
        n_frames_used = len(lig_pos_frames)
        print(f"  {resid:<8} {resname:<8} {n_close/n_frames_used:>12.1f}       {n_medium/n_frames_used:.1f}")

print("\n── Done ────────────────────────────────────────────────────────────────")
