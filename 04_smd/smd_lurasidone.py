#!/usr/bin/env python
"""
SMD (Steered Molecular Dynamics) — Lurasidone / GIPR
Constant-velocity pulling via harmonic spring on ligand COM
Force field : same as production (ff14SB + GAFF2 + TIP3P, HMR 4 fs)

Usage:
    python smd_lurasidone.py --test       # 0.5 ns benchmark → estimate time
    python smd_lurasidone.py              # full run (5 ns, 5 replicates)
    python smd_lurasidone.py --rep 2      # single replicate #2
"""

import sys, os, time, argparse, json
import numpy as np
from pathlib import Path

import openmm as mm
import openmm.app as app
import openmm.unit as unit
import parmed as pmd

# ── Config ─────────────────────────────────────────────────────────────────────
WORK_DIR    = Path("/home/stalin/Downloads/Enamine/GIPR/FDA/md/production/Lurasidone")
SMD_DIR     = Path("/home/stalin/Downloads/Enamine/GIPR/FDA/md/smd_Lurasidone")

DT_FS       = 4.0                     # femtoseconds (HMR)
PULL_NS     = 5.0                     # pulling duration per replicate
PULL_VEL    = 0.5                     # Å/ns  (slow → accurate; use 5 for quick screen)
K_SPRING    = 1000.0                  # kJ/mol/nm²
N_REPS      = 5                       # number of independent replicates
RECORD_EVERY = 500                    # record force every N steps (~2 ps)

TEST_NS     = 0.5                     # duration for benchmark test run

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--test",  action="store_true", help="0.5 ns benchmark only")
parser.add_argument("--rep",   type=int, default=None, help="run single replicate N")
args = parser.parse_args()

SMD_DIR.mkdir(parents=True, exist_ok=True)

# ── Load topology ──────────────────────────────────────────────────────────────
prmtop = WORK_DIR / "complex.prmtop"
inpcrd = WORK_DIR / "complex.inpcrd"
struct = pmd.load_file(str(prmtop), str(inpcrd))
print(f"  Topology : {len(struct.atoms):,} atoms")

# ── Identify ligand and protein atom indices ────────────────────────────────────
lig_indices  = [i for i, a in enumerate(struct.atoms) if a.residue.name == "LIG"]
prot_indices = [i for i, a in enumerate(struct.atoms)
                if a.residue.name not in ("LIG", "WAT", "HOH", "Na+", "Cl-", "SOD", "CLA")]

if not lig_indices:
    sys.exit("ERROR: no LIG residue found in topology.")

print(f"  Ligand   : {len(lig_indices)} atoms  (res LIG)")
print(f"  Protein  : {len(prot_indices)} atoms")

# ── Build OpenMM system ────────────────────────────────────────────────────────
def make_sim(extra_force=None) -> app.Simulation:
    system = struct.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        hydrogenMass=1.5 * unit.dalton,
    )
    system.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 25))
    if extra_force is not None:
        system.addForce(extra_force)
    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, DT_FS * unit.femtosecond
    )
    platform   = mm.Platform.getPlatformByName("CUDA")
    props      = {"DeviceIndex": "0", "Precision": "mixed"}
    sim = app.Simulation(struct.topology, system, integrator, platform, props)
    sim.loadState(str(WORK_DIR / "equilibrated.xml"))
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)
    return sim

# ── Get initial COM positions ──────────────────────────────────────────────────
def get_com(sim, indices):
    state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    pos   = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    masses = np.array([struct.atoms[i].mass for i in indices])
    coords = pos[indices]
    return np.average(coords, axis=0, weights=masses)  # nm

# ── SMD pulling force ─────────────────────────────────────────────────────────
def make_pull_force(anchor_nm):
    """
    Harmonic spring on ligand COM.
    E = 0.5 * k * [(x1-rx)² + (y1-ry)² + (z1-rz)²]
    anchor (rx,ry,rz) is updated each step to move at constant velocity.
    """
    expr  = "0.5*k*((x1-rx)^2 + (y1-ry)^2 + (z1-rz)^2)"
    force = mm.CustomCentroidBondForce(1, expr)
    force.addGlobalParameter("k",  K_SPRING)
    force.addGlobalParameter("rx", anchor_nm[0])
    force.addGlobalParameter("ry", anchor_nm[1])
    force.addGlobalParameter("rz", anchor_nm[2])
    grp = force.addGroup(lig_indices)
    force.addBond([grp], [])
    force.setForceGroup(1)
    return force

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK / TEST RUN
# ══════════════════════════════════════════════════════════════════════════════
if args.test:
    print(f"\n── Benchmark run ({TEST_NS} ns) ─────────────────────────────")
    sim   = make_sim()
    steps = int(TEST_NS * 1_000_000 / DT_FS)
    t0    = time.perf_counter()
    sim.step(steps)
    elapsed  = time.perf_counter() - t0
    ns_day   = TEST_NS / elapsed * 86_400
    full_run = PULL_NS * N_REPS       # total ns for all replicates
    eta_h    = full_run / ns_day * 24

    print(f"\n  Benchmark result:")
    print(f"    Speed        : {ns_day:.0f} ns/day")
    print(f"    Per replicate: {PULL_NS:.1f} ns  →  {PULL_NS/ns_day*24*60:.1f} min")
    print(f"    Full SMD     : {N_REPS} × {PULL_NS} ns  →  {eta_h*60:.0f} min  ({eta_h:.2f} h)")
    print(f"\n  Run full SMD with:  python smd_lurasidone.py")
    sys.exit(0)

# ══════════════════════════════════════════════════════════════════════════════
# FULL SMD REPLICATE(S)
# ══════════════════════════════════════════════════════════════════════════════
reps_to_run = [args.rep] if args.rep else list(range(1, N_REPS + 1))
total_steps = int(PULL_NS * 1_000_000 / DT_FS)
# velocity in nm/step
vel_nm_step = PULL_VEL * 0.1 / 1e6 * DT_FS   # Å/ns → nm/step (Å*0.1=nm, /1e6 ns/fs, *DT_FS fs/step)

print(f"\n  Pull velocity  : {PULL_VEL} Å/ns")
print(f"  Spring const   : {K_SPRING} kJ/mol/nm²")
print(f"  Duration/rep   : {PULL_NS} ns  ({total_steps:,} steps)")
print(f"  Replicates     : {reps_to_run}\n")

all_work = []

for rep in reps_to_run:
    out_dir = SMD_DIR / f"rep{rep:02d}"
    out_dir.mkdir(exist_ok=True)

    print(f"── Replicate {rep} ──────────────────────────────────────────")

    # Initial geometry (use production end state + re-randomise velocities)
    pull_force = make_pull_force(np.zeros(3))   # placeholder anchor
    sim        = make_sim(extra_force=pull_force)

    lig_com0  = get_com(sim, lig_indices)
    prot_com0 = get_com(sim, prot_indices)

    # Pulling direction: ligand COM → away from protein COM (unit vector)
    pull_vec = lig_com0 - prot_com0
    pull_vec /= np.linalg.norm(pull_vec)

    anchor    = lig_com0.copy()   # starts at ligand COM
    force_idx = next(i for i in range(sim.system.getNumForces())
                     if isinstance(sim.system.getForce(i), mm.CustomCentroidBondForce))
    pull_f    = sim.system.getForce(force_idx)

    sim.context.setParameter("rx", float(anchor[0]))
    sim.context.setParameter("ry", float(anchor[1]))
    sim.context.setParameter("rz", float(anchor[2]))

    print(f"  Lig COM₀   : {lig_com0} nm")
    print(f"  Pull dir   : {pull_vec}")

    # Output arrays
    times_ps, extensions_nm, forces_kj = [], [], []

    # Step in RECORD_EVERY chunks — anchor updated per chunk (negligible discretization)
    n_chunks  = total_steps // RECORD_EVERY
    t0 = time.perf_counter()
    for chunk_i in range(n_chunks):
        step = chunk_i * RECORD_EVERY

        # Advance anchor by chunk displacement
        anchor += pull_vec * vel_nm_step * RECORD_EVERY
        sim.context.setParameter("rx", float(anchor[0]))
        sim.context.setParameter("ry", float(anchor[1]))
        sim.context.setParameter("rz", float(anchor[2]))

        sim.step(RECORD_EVERY)

        lig_com   = get_com(sim, lig_indices)
        ext       = float(np.dot(lig_com - lig_com0, pull_vec))   # nm
        dr        = lig_com - anchor
        force_kj  = K_SPRING * np.linalg.norm(dr)                 # kJ/mol/nm × nm
        t_ps      = step * DT_FS / 1000.0

        times_ps.append(t_ps)
        extensions_nm.append(ext)
        forces_kj.append(force_kj)

        if chunk_i % 100 == 0:
            elapsed = time.perf_counter() - t0
            ns_done = step * DT_FS / 1e6
            ns_day  = ns_done / elapsed * 86_400 if elapsed > 0 else 0
            eta_min = (PULL_NS - ns_done) / ns_day * 1440 if ns_day > 0 else 0
            print(f"  {ns_done:.2f}/{PULL_NS} ns  "
                  f"F={force_kj:.1f} kJ/mol/nm  "
                  f"{ns_day:.0f} ns/day  ETA {eta_min:.0f} min", flush=True)

    # Save results
    results = {
        "replicate":       rep,
        "pull_velocity_A_per_ns": PULL_VEL,
        "k_spring_kJ_mol_nm2":    K_SPRING,
        "pull_direction":  pull_vec.tolist(),
        "lig_com0_nm":     lig_com0.tolist(),
        "time_ps":         times_ps,
        "extension_nm":    extensions_nm,
        "force_kJ_mol_nm": forces_kj,
    }
    out_json = out_dir / "smd_force_extension.json"
    out_json.write_text(json.dumps(results, indent=2))

    # Work via trapezoidal integration  W = ∫F·dx
    ext_arr = np.array(extensions_nm)
    frc_arr = np.array(forces_kj)
    work_kj = np.trapz(frc_arr, ext_arr)
    work_kcal = work_kj / 4.184
    all_work.append(work_kj)

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed/60:.1f} min")
    print(f"  Max force : {max(forces_kj):.1f} kJ/mol/nm  "
          f"({max(forces_kj)/4.184:.1f} kcal/mol/Å)")
    print(f"  Work (W)  : {work_kj:.1f} kJ/mol  ({work_kcal:.1f} kcal/mol)\n")

# ── Summary ────────────────────────────────────────────────────────────────────
if len(all_work) > 1:
    w  = np.array(all_work)
    print(f"{'='*55}")
    print(f"  SMD Summary — Lurasidone ({len(all_work)} replicates)")
    print(f"  Mean work : {w.mean():.1f} ± {w.std():.1f} kJ/mol")
    print(f"  Range     : {w.min():.1f} – {w.max():.1f} kJ/mol")
    print(f"{'='*55}")

    summary = {"compound": "Lurasidone", "n_reps": len(all_work),
               "mean_work_kJ_mol": float(w.mean()),
               "std_work_kJ_mol":  float(w.std()),
               "works_kJ_mol":     w.tolist()}
    (SMD_DIR / "smd_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Saved → {SMD_DIR / 'smd_summary.json'}")
