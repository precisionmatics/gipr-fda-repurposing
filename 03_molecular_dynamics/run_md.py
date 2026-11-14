#!/usr/bin/env python
"""
Production MD pipeline — GIPR + ligand complex, 100 ns NPT
Force field : ff14SB (protein) + GAFF2 (ligand) + TIP3P (water)
Timestep    : 4 fs with hydrogen mass repartitioning
Checkpoint  : every 200 ps  →  max 200 ps lost on power cut
Resumable   : re-run the same command after power loss to continue

Usage:
    python run_md.py Lurasidone
    python run_md.py Montelukast
    python run_md.py Risperidone
    python run_md.py Atovaquone
"""

import sys, os, time, subprocess
from pathlib import Path

import openmm as mm
import openmm.app as app
import openmm.unit as unit
import parmed as pmd

# ── Config ─────────────────────────────────────────────────────────────────────
COMPOUNDS = {
    "Lurasidone":  0,
    "Montelukast":  0,   # neutral (protonated COOH) form for GAFF2 BCC
    "Risperidone": 0,
    "Atovaquone":  0,
}

MD_BASE     = Path("/home/stalin/Downloads/Enamine/GIPR/FDA/md")
TOTAL_NS    = 100.0
DT_FS       = 4.0
TOTAL_STEPS = int(TOTAL_NS * 1_000_000 / DT_FS)  # 25,000,000

CHUNK_STEPS  = 50_000    # save checkpoint every 200 ps
TRAJ_EVERY   = 2_500     # write frame every 10 ps
LOG_EVERY    = 25_000    # write log every 100 ps
NVT_STEPS    = 250_000   # 1 ns NVT equilibration
NPT_STEPS    = 250_000   # 1 ns NPT equilibration

# ── Helpers ────────────────────────────────────────────────────────────────────
def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr[-700:]}")
    return r

def read_steps(work: Path) -> int:
    f = work / "steps_done.txt"
    return int(f.read_text().strip()) if f.exists() else 0

def save_steps(work: Path, n: int):
    tmp = work / ".steps_tmp"
    tmp.write_text(str(n))
    tmp.rename(work / "steps_done.txt")   # atomic rename

def make_sim(struct, barostat: bool) -> app.Simulation:
    system = struct.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        hydrogenMass=1.5 * unit.dalton,
    )
    if barostat:
        system.addForce(
            mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 25)
        )
    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, DT_FS * unit.femtosecond
    )
    platform   = mm.Platform.getPlatformByName("CUDA")
    properties = {"DeviceIndex": "0", "Precision": "mixed"}
    return app.Simulation(struct.topology, system, integrator, platform, properties)

# ── Entry point ────────────────────────────────────────────────────────────────
if len(sys.argv) < 2 or sys.argv[1] not in COMPOUNDS:
    print(f"Usage: python run_md.py <compound>  ({' | '.join(COMPOUNDS)})")
    sys.exit(1)

name   = sys.argv[1]
charge = COMPOUNDS[name]
pdb_in = MD_BASE / f"{name}_GIPR.pdb"
work   = MD_BASE / "production" / name
work.mkdir(parents=True, exist_ok=True)

prmtop = work / "complex.prmtop"
inpcrd = work / "complex.inpcrd"

print(f"\n{'='*58}")
print(f"  {name}  |  ligand charge {charge:+d}  |  {TOTAL_NS:.0f} ns NPT")
print(f"  Work dir : {work}")
print(f"{'='*58}\n")

# ═══════════════════════════════════════════════════════════════
# STAGE 0 — System setup (runs once, skipped on resume)
# ═══════════════════════════════════════════════════════════════
if not (work / "setup_done.flag").exists():
    print("── [0/4] System setup ──────────────────────────────────")

    # Split protein / ligand
    prot, lig = [], []
    with open(pdb_in) as fh:
        for ln in fh:
            if ln.startswith(("ATOM", "TER")):  prot.append(ln)
            elif ln.startswith("HETATM"):        lig.append(ln)
    (work / "protein.pdb").write_text("".join(prot) + "END\n")
    (work / "ligand.pdb").write_text("".join(lig)   + "END\n")

    # Fix protein for AMBER
    sh(["pdb4amber",
        "-i", str(work / "protein.pdb"),
        "-o", str(work / "protein_fixed.pdb"),
        "--nohyd"])
    print("  pdb4amber      OK")

    # Add H atoms with RDKit before antechamber (docking PDBs are heavy-atom only)
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromPDBFile(str(work / "ligand.pdb"), removeHs=False, sanitize=True)
    if mol is None:
        mol = Chem.MolFromPDBFile(str(work / "ligand.pdb"), removeHs=False, sanitize=False)
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    mol_h = Chem.AddHs(mol, addCoords=True)
    AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3()) if mol_h.GetNumConformers() == 0 else None
    Chem.MolToPDBFile(mol_h, str(work / "ligand_h.pdb"))
    print("  RDKit H-add    OK")

    # GAFF2 parameters for ligand (BCC charges, ~5 min)
    print("  antechamber    running GAFF2 BCC charges (~5 min)...")
    sh(["antechamber",
        "-i",  str(work / "ligand_h.pdb"), "-fi", "pdb",
        "-o",  str(work / "ligand.mol2"), "-fo", "mol2",
        "-c",  "bcc", "-s", "2", "-at", "gaff2",
        "-nc", str(charge)], cwd=work)
    sh(["parmchk2",
        "-i", str(work / "ligand.mol2"), "-f", "mol2",
        "-o", str(work / "ligand.frcmod")], cwd=work)
    print("  antechamber    OK  (GAFF2 mol2 + frcmod ready)")

    # Build solvated system with tleap
    print("  tleap          solvating with TIP3P (10 Å padding)...")
    tleap_script = (
        f"source leaprc.protein.ff14SB\n"
        f"source leaprc.gaff2\n"
        f"source leaprc.water.tip3p\n"
        f"LIG  = loadmol2 {work}/ligand.mol2\n"
        f"loadamberparams {work}/ligand.frcmod\n"
        f"PROT = loadpdb {work}/protein_fixed.pdb\n"
        f"SYS  = combine {{PROT LIG}}\n"
        f"solvatebox SYS TIP3PBOX 10.0\n"
        f"addions SYS Na+ 0\n"
        f"addions SYS Cl- 0\n"
        f"saveamberparm SYS {prmtop} {inpcrd}\n"
        f"quit\n"
    )
    (work / "tleap.in").write_text(tleap_script)
    r = subprocess.run(["tleap", "-f", str(work / "tleap.in")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"tleap failed:\n{r.stderr[-900:]}")

    # Report system info from tleap output
    for ln in r.stdout.splitlines():
        if any(k in ln for k in ("total atoms", "Total", "Added", "vdw box")):
            print(" ", ln.strip())

    # Quick sanity: topology file must exist and be non-empty
    if not prmtop.exists() or prmtop.stat().st_size < 1000:
        raise RuntimeError("tleap did not produce a valid prmtop — check tleap.in")

    (work / "setup_done.flag").touch()
    print("  Setup COMPLETE.\n")
else:
    print("── [0/4] Setup          already done — skipping.\n")

# ── Load topology ──────────────────────────────────────────────────────────────
struct  = pmd.load_file(str(prmtop), str(inpcrd))
n_atoms = len(struct.atoms)
print(f"  Topology loaded : {n_atoms:,} atoms\n")

# ═══════════════════════════════════════════════════════════════
# STAGE 1 — Energy minimisation
# ═══════════════════════════════════════════════════════════════
min_xml = work / "minimized.xml"
if not min_xml.exists():
    print("── [1/4] Minimisation ──────────────────────────────────")
    sim = make_sim(struct, barostat=False)
    sim.context.setPositions(struct.positions)
    sim.minimizeEnergy(maxIterations=2000)
    state = sim.context.getState(getEnergy=True, enforcePeriodicBox=True)
    pe = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    print(f"  Potential energy : {pe:,.1f} kcal/mol")
    if pe > 0:
        print("  WARNING: positive PE after minimisation — check for clashes")
    sim.saveState(str(min_xml))
    print("  Minimisation COMPLETE.\n")
else:
    print("── [1/4] Minimisation   already done — skipping.\n")

# ═══════════════════════════════════════════════════════════════
# STAGE 2 — Equilibration (NVT + NPT)
# ═══════════════════════════════════════════════════════════════
equil_xml = work / "equilibrated.xml"
if not equil_xml.exists():
    print("── [2/4] Equilibration ─────────────────────────────────")

    # NVT — heat system to 300 K
    print(f"  NVT {NVT_STEPS * DT_FS / 1e6:.1f} ns ...")
    sim = make_sim(struct, barostat=False)
    sim.loadState(str(min_xml))
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)
    sim.reporters.append(app.StateDataReporter(
        str(work / "equil_nvt.log"), 25_000,
        step=True, temperature=True, potentialEnergy=True,
        progress=True, totalSteps=NVT_STEPS))
    sim.step(NVT_STEPS)
    nvt_xml = work / "nvt_done.xml"
    sim.saveState(str(nvt_xml))
    print("  NVT done.")

    # NPT — equilibrate density / box
    print(f"  NPT {NPT_STEPS * DT_FS / 1e6:.1f} ns ...")
    sim = make_sim(struct, barostat=True)
    sim.loadState(str(nvt_xml))
    sim.reporters.append(app.StateDataReporter(
        str(work / "equil_npt.log"), 25_000,
        step=True, temperature=True, density=True,
        progress=True, totalSteps=NPT_STEPS))
    sim.step(NPT_STEPS)
    sim.saveState(str(equil_xml))
    print("  Equilibration COMPLETE.\n")
else:
    print("── [2/4] Equilibration  already done — skipping.\n")

# ═══════════════════════════════════════════════════════════════
# STAGE 3 — Production (100 ns, resumable)
# ═══════════════════════════════════════════════════════════════
steps_done = read_steps(work)
if steps_done >= TOTAL_STEPS:
    print(f"\n  ✓  {name} — 100 ns already complete!")
    sys.exit(0)

chk_file  = work / "production.chk"
seg_num   = len(list(work.glob("traj_*.dcd"))) + 1
traj_file = work / f"traj_{seg_num:03d}.dcd"
log_file  = work / f"prod_{seg_num:03d}.log"

print("── [3/4] Production ────────────────────────────────────")
print(f"  Segment    : {seg_num}")
print(f"  Progress   : {steps_done * DT_FS / 1e6:.1f} / {TOTAL_NS:.0f} ns  "
      f"({100 * steps_done / TOTAL_STEPS:.1f}%)")
print(f"  Trajectory : {traj_file.name}")
print(f"  Checkpoint : every {CHUNK_STEPS * DT_FS / 1e6:.0f} ps\n")

sim = make_sim(struct, barostat=True)

if chk_file.exists() and steps_done > 0:
    sim.loadCheckpoint(str(chk_file))
    print(f"  Checkpoint loaded — resuming from "
          f"{steps_done * DT_FS / 1e6:.1f} ns\n")
else:
    sim.loadState(str(equil_xml))
    print("  Starting fresh production run.\n")

sim.reporters.append(app.DCDReporter(str(traj_file), TRAJ_EVERY))
sim.reporters.append(app.StateDataReporter(
    str(log_file), LOG_EVERY,
    step=True, time=True, temperature=True,
    density=True, potentialEnergy=True, speed=True))

t0 = time.perf_counter()

while steps_done < TOTAL_STEPS:
    chunk = min(CHUNK_STEPS, TOTAL_STEPS - steps_done)
    sim.step(chunk)
    steps_done += chunk

    # Atomic checkpoint — safe even if power dies here
    sim.saveCheckpoint(str(chk_file))
    save_steps(work, steps_done)

    ns_done = steps_done * DT_FS / 1_000_000
    elapsed = time.perf_counter() - t0
    ns_day  = ns_done * 86_400 / elapsed if elapsed > 0 else 0
    eta_h   = (TOTAL_NS - ns_done) / ns_day * 24 if ns_day > 0 else 0
    print(f"  {ns_done:6.1f}/{TOTAL_NS:.0f} ns  "
          f"({100*steps_done/TOTAL_STEPS:5.1f}%)  "
          f"{ns_day:5.0f} ns/day  "
          f"ETA {eta_h:.1f} h", flush=True)

print(f"\n  ✓  {name} — 100 ns COMPLETE!")
print(f"  Trajectories : {sorted(work.glob('traj_*.dcd'))}\n")
