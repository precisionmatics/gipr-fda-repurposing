#!/usr/bin/env python3
"""Quick RMSD + COM drift + contact stability check."""
import numpy as np
import MDAnalysis as mda
import sys, warnings
warnings.filterwarnings('ignore')

compound = sys.argv[1]
WORK = f"/home/stalin/Downloads/Enamine/GIPR/FDA/md/production/{compound}"
import glob
trajs = sorted(glob.glob(f"{WORK}/traj_*.dcd"))

u = mda.Universe(f"{WORK}/complex.prmtop", trajs)
n_frames = len(u.trajectory)
lig  = u.select_atoms("resname LIG")
prot = u.select_atoms("protein")

u.trajectory[0]
ref_lig_pos  = lig.positions.copy()
ref_ca_pos   = u.select_atoms("protein and name CA").positions.copy()
ref_ca_mean  = ref_ca_pos.mean(0)
ref_lig_c    = ref_lig_pos - ref_ca_mean
com_init     = lig.center_of_mass().copy()

rmsd_list = []
com_drift = []
step = 10

for i, ts in enumerate(u.trajectory):
    if i % step != 0:
        continue
    ca_now = u.select_atoms("protein and name CA").positions
    mob = ca_now - ca_now.mean(0)
    ref = ref_ca_pos - ref_ca_mean
    H = mob.T @ ref
    U2, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U2.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U2.T
    lig_aligned = (lig.positions - ca_now.mean(0)) @ R
    rmsd = np.sqrt(((lig_aligned - ref_lig_c)**2).mean())
    rmsd_list.append(rmsd)
    com_drift.append(np.linalg.norm(lig.center_of_mass() - com_init))

rmsd_arr = np.array(rmsd_list)
drift_arr = np.array(com_drift)
q = len(rmsd_arr) // 4

print(f"{compound}:")
print(f"  RMSD  0-25ns: {rmsd_arr[:q].mean():.2f} Å  |  25-50ns: {rmsd_arr[q:2*q].mean():.2f} Å  |  50-75ns: {rmsd_arr[2*q:3*q].mean():.2f} Å  |  75-100ns: {rmsd_arr[3*q:].mean():.2f} Å")
print(f"  Max RMSD: {rmsd_arr.max():.2f} Å  |  Max COM drift: {drift_arr.max():.2f} Å  |  Late COM drift: {drift_arr[3*q:].mean():.2f} Å")

late_rmsd  = rmsd_arr[3*q:].mean()
max_drift  = drift_arr.max()
if late_rmsd < 2.5 and max_drift < 5.0:
    verdict = "STABLE"
elif late_rmsd < 4.0 and max_drift < 8.0:
    verdict = "MODERATE"
else:
    verdict = "UNSTABLE/ESCAPED"
print(f"  Verdict: {verdict}")
