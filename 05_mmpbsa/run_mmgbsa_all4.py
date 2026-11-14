#!/usr/bin/env python3
"""
MM-GBSA pipeline for all 4 GIPR consensus hits — 100 ns trajectories.
Uses: ante-MMPBSA.py (stripped topologies) + cpptraj (strip solvent) + MMPBSA.py (igb=5)
Skips first 10 ns; samples every 10th frame from remaining 90 ns → 900 frames per compound.
"""
import os, subprocess, json
from pathlib import Path

AMBER_BIN  = Path("/home/stalin/miniforge3/envs/unomd/bin")
AMBERHOME  = str(AMBER_BIN.parent)
PROD_BASE  = Path("/run/media/stalin/Expansion/Sandisk_ssd/FDA/md/production")
OUT_BASE   = Path("/run/media/stalin/Expansion/Sandisk_ssd/FDA/md/mmgbsa")
OUT_BASE.mkdir(exist_ok=True)

import os as _os
AMBER_ENV  = {**_os.environ, "AMBERHOME": AMBERHOME,
              "PATH": f"{AMBER_BIN}:{_os.environ.get('PATH','')}"}

COMPOUNDS = ["Montelukast", "Lurasidone", "Risperidone", "Atovaquone"]

# 100 ns @ 10 ps/frame = 10000 frames; skip first 1000 (10 ns); every 10th = 900 frames
STARTFRAME = 1001
ENDFRAME   = 10000
INTERVAL   = 10

MMPBSA_IN = """\
Per-frame MM-PBSA: PB implicit solvent, salt=0.15 M, skip 10 ns equilibration
&general
  startframe={start}, endframe={end}, interval={interval},
  verbose=2, keep_files=0,
/
&pb
  istrng=150, radiopt=0, inp=2,
/
"""

def run(cmd, cwd=None, label=""):
    print(f"  [{label}] {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd,
                       capture_output=True, text=True, env=AMBER_ENV)
    if r.returncode != 0:
        print(f"  STDERR: {r.stderr[-600:]}")
        raise RuntimeError(f"{label} failed (rc={r.returncode})")
    return r.stdout

results = {}

for cpd in COMPOUNDS:
    print(f"\n{'='*60}")
    print(f"  {cpd}")
    print(f"{'='*60}")

    src   = PROD_BASE / cpd
    work  = OUT_BASE / cpd
    work.mkdir(exist_ok=True)

    prmtop = src / "complex.prmtop"
    trajs  = sorted(src.glob("traj_*.dcd"))

    # ── 1. Strip solvent with cpptraj ──────────────────────────────────────
    strip_nc  = work / "traj_strip.nc"
    cpptraj_in = work / "strip.cpptraj"
    lines = [f"parm {prmtop}"]
    for t in trajs:
        lines.append(f"trajin {t}")
    lines += [
        "strip :WAT,Na+,Cl-",
        f"trajout {strip_nc} netcdf",
        "run",
        "quit",
    ]
    cpptraj_in.write_text("\n".join(lines))

    if not strip_nc.exists():
        run([AMBER_BIN/"cpptraj", "-i", cpptraj_in], cwd=work, label="cpptraj strip")
        print(f"  Stripped trajectory → {strip_nc.name}")
    else:
        print(f"  Stripped trajectory already exists — skipping.")

    # ── 2. Create stripped topologies with ante-MMPBSA.py ─────────────────
    com_prmtop = work / "com.prmtop"
    rec_prmtop = work / "rec.prmtop"
    lig_prmtop = work / "lig.prmtop"

    if not com_prmtop.exists():
        run([AMBER_BIN/"ante-MMPBSA.py",
             "-p", prmtop,
             "-c", com_prmtop,
             "-r", rec_prmtop,
             "-l", lig_prmtop,
             "-s", ":WAT,Na+,Cl-",
             "-n", ":LIG",
             "--radii", "mbondi2"],
            cwd=work, label="ante-MMPBSA.py")
        print("  Stripped topologies created.")
    else:
        print("  Stripped topologies already exist — skipping.")

    # ── 3. Write mmpbsa.in ─────────────────────────────────────────────────
    mmpbsa_in = work / "mmpbsa.in"
    mmpbsa_in.write_text(MMPBSA_IN.format(
        start=STARTFRAME, end=ENDFRAME, interval=INTERVAL))

    # ── 4. Run MMPBSA.py ───────────────────────────────────────────────────
    results_dat = work / "MMPBSA_pb_results.dat"
    energy_csv  = work / "MMPBSA_pb_energy.csv"

    if not results_dat.exists():
        run([AMBER_BIN/"MMPBSA.py", "-O",
             "-i",  mmpbsa_in,
             "-o",  results_dat,
             "-eo", energy_csv,
             "-cp", com_prmtop,
             "-rp", rec_prmtop,
             "-lp", lig_prmtop,
             "-y",  strip_nc],
            cwd=work, label="MMPBSA.py")
        print(f"  MMPBSA complete → {results_dat.name}")
    else:
        print(f"  MMPBSA results already exist — skipping.")

    # ── 5. Parse result ────────────────────────────────────────────────────
    if results_dat.exists():
        txt = results_dat.read_text()
        import re
        # Look for "DELTA TOTAL" line
        m = re.search(r'DELTA TOTAL\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', txt)
        if m:
            dg_mean = float(m.group(1))
            dg_std  = float(m.group(3))
            results[cpd] = {"dG_kcal": dg_mean, "dG_std": dg_std}
            print(f"\n  ★ {cpd}  ΔG_bind = {dg_mean:.2f} ± {dg_std:.2f} kcal/mol")
        else:
            print("  Could not parse DELTA TOTAL from results.")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n\n{'='*60}")
print("  MM-GBSA SUMMARY (100 ns, igb=5)")
print(f"{'='*60}")
print(f"  {'Compound':<14} {'ΔG_bind (kcal/mol)':<25} {'Rank'}")
ranked = sorted(results.items(), key=lambda x: x[1]['dG_kcal'])
for rank, (cpd, v) in enumerate(ranked, 1):
    print(f"  {cpd:<14} {v['dG_kcal']:>8.2f} ± {v['dG_std']:.2f}       {rank}")

# Save JSON
out_json = OUT_BASE / "mmgbsa_100ns_results.json"
out_json.write_text(json.dumps(results, indent=2))
print(f"\nResults saved → {out_json}")
