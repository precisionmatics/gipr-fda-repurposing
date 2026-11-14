#!/usr/bin/env python3
"""
MM-GBSA for Lurasidone 500 ns trajectory.
Skips first 10 ns; samples every 10th frame from remaining 490 ns → 4900 frames.
Run this after extend_md_500ns.py completes.
"""
import os, subprocess, json, re
from pathlib import Path

AMBER_BIN = Path("/home/stalin/miniforge3/envs/unomd/bin")
AMBER_ENV = {**os.environ, "AMBERHOME": str(AMBER_BIN.parent),
             "PATH": f"{AMBER_BIN}:{os.environ.get('PATH', '')}"}
SRC   = Path("/run/media/stalin/Expansion/Sandisk_ssd/FDA/md/production/Lurasidone")
WORK  = Path("/run/media/stalin/Expansion/Sandisk_ssd/FDA/md/mmgbsa/Lurasidone_500ns")
WORK.mkdir(parents=True, exist_ok=True)

# 500 ns @ 10 ps/frame = 50000 frames; skip first 1000 (10 ns); every 10th → 4900 frames
STARTFRAME = 1001
ENDFRAME   = 50000
INTERVAL   = 10

MMPBSA_IN = f"""\
MM-PBSA Lurasidone 500 ns: PB implicit solvent, salt=0.15 M, skip 10 ns
&general
  startframe={STARTFRAME}, endframe={ENDFRAME}, interval={INTERVAL},
  verbose=2, keep_files=0,
/
&pb
  istrng=150, radiopt=0, inp=2,
/
"""

def run(cmd, cwd=None, label=""):
    print(f"  [{label}] running...")
    r = subprocess.run([str(c) for c in cmd], cwd=cwd,
                       capture_output=True, text=True, env=AMBER_ENV)
    if r.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{r.stderr[-600:]}")
    return r.stdout

prmtop = SRC / "complex.prmtop"
trajs  = sorted(SRC.glob("traj_*.dcd"))
print(f"Trajectories: {[t.name for t in trajs]}")

# 1. Strip with cpptraj
strip_nc   = WORK / "traj_strip.nc"
cpptraj_in = WORK / "strip.cpptraj"
lines = [f"parm {prmtop}"]
for t in trajs:
    lines.append(f"trajin {t}")
lines += ["strip :WAT,Na+,Cl-", f"trajout {strip_nc} netcdf", "run", "quit"]
cpptraj_in.write_text("\n".join(lines))

if not strip_nc.exists():
    run([AMBER_BIN/"cpptraj", "-i", cpptraj_in], cwd=WORK, label="cpptraj strip")

# 2. Stripped topologies
com_p = WORK / "com.prmtop"
rec_p = WORK / "rec.prmtop"
lig_p = WORK / "lig.prmtop"
if not com_p.exists():
    run([AMBER_BIN/"ante-MMPBSA.py",
         "-p", prmtop, "-c", com_p, "-r", rec_p, "-l", lig_p,
         "-s", ":WAT,Na+,Cl-", "-n", ":LIG", "--radii", "mbondi2"],
        cwd=WORK, label="ante-MMPBSA.py")

# 3. Run MMPBSA
(WORK / "mmpbsa.in").write_text(MMPBSA_IN)
results_dat = WORK / "MMPBSA_results.dat"
if not results_dat.exists():
    run([AMBER_BIN/"MMPBSA.py", "-O",
         "-i", WORK/"mmpbsa.in", "-o", results_dat, "-eo", WORK/"MMPBSA_energy.csv",
         "-cp", com_p, "-rp", rec_p, "-lp", lig_p, "-y", strip_nc],
        cwd=WORK, label="MMPBSA.py")

# 4. Parse
txt = results_dat.read_text()
m = re.search(r'DELTA TOTAL\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', txt)
if m:
    print(f"\n★ Lurasidone 500 ns  ΔG_bind = {float(m.group(1)):.2f} ± {float(m.group(3)):.2f} kcal/mol")
