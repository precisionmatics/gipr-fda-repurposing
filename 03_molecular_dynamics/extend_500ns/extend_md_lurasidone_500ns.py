#!/usr/bin/env python3
"""
Extend Lurasidone MD from 100 ns → 500 ns.
Resumes from production.chk — no re-equilibration needed.
Writes traj_003.dcd, traj_004.dcd, ... in the same production dir.
"""
import time
from pathlib import Path
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import parmed as pmd

WORK   = Path("/run/media/stalin/Expansion/Sandisk_ssd/FDA/md/production/Lurasidone")
DT_FS  = 4.0
TOTAL_NS    = 500.0
TOTAL_STEPS = int(TOTAL_NS * 1_000_000 / DT_FS)   # 125,000,000

# Read actual checkpoint position from steps_done.txt
_steps_file = WORK / "steps_done.txt"
STEPS_ALREADY = int(_steps_file.read_text().strip()) if _steps_file.exists() else 25_000_000
REMAIN_STEPS  = TOTAL_STEPS - STEPS_ALREADY

CHUNK_STEPS = 50_000      # checkpoint every 200 ps
TRAJ_EVERY  = 2_500       # frame every 10 ps
LOG_EVERY   = 25_000      # log every 100 ps

def save_steps(n: int):
    tmp = WORK / ".steps_tmp"
    tmp.write_text(str(n))
    tmp.rename(WORK / "steps_done.txt")

def make_sim():
    struct = pmd.load_file(str(WORK / "complex.prmtop"), str(WORK / "complex.inpcrd"))
    system = struct.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        hydrogenMass=1.5 * unit.dalton,
    )
    system.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 25))
    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, DT_FS * unit.femtosecond
    )
    platform   = mm.Platform.getPlatformByName("CUDA")
    properties = {"DeviceIndex": "0", "Precision": "mixed"}
    return app.Simulation(struct.topology, system, integrator, platform, properties), struct

print(f"\n{'='*60}")
print(f"  Lurasidone  |  extending 100 → 500 ns  ({REMAIN_STEPS/1e6:.0f} M steps)")
print(f"{'='*60}\n")

chk_file = WORK / "production.chk"
if not chk_file.exists():
    raise FileNotFoundError(f"Checkpoint not found: {chk_file}")

sim, struct = make_sim()
sim.loadCheckpoint(str(chk_file))
print(f"  Checkpoint loaded from {chk_file.name}\n")

seg_num     = len(list(WORK.glob("traj_*.dcd"))) + 1
steps_done  = STEPS_ALREADY

traj_file = WORK / f"traj_{seg_num:03d}.dcd"
log_file  = WORK / f"prod_{seg_num:03d}.log"

sim.reporters.append(app.DCDReporter(str(traj_file), TRAJ_EVERY))
sim.reporters.append(app.StateDataReporter(
    str(log_file), LOG_EVERY,
    step=True, time=True, temperature=True,
    density=True, potentialEnergy=True, speed=True))

print(f"  Writing trajectory  : {traj_file.name}")
print(f"  Writing log         : {log_file.name}")
print(f"  Checkpoint every    : 200 ps")
print(f"  Frames every        : 10 ps\n")

t0 = time.perf_counter()

while steps_done < TOTAL_STEPS:
    chunk       = min(CHUNK_STEPS, TOTAL_STEPS - steps_done)
    sim.step(chunk)
    steps_done += chunk

    sim.saveCheckpoint(str(chk_file))
    save_steps(steps_done)

    ns_done  = steps_done * DT_FS / 1_000_000
    elapsed  = time.perf_counter() - t0
    ns_day   = (steps_done - STEPS_ALREADY) * DT_FS / 1_000_000 / elapsed * 86_400
    eta_h    = (TOTAL_NS - ns_done) / ns_day * 24 if ns_day > 0 else 0
    print(f"  {ns_done:6.1f}/{TOTAL_NS:.0f} ns  "
          f"({100*ns_done/TOTAL_NS:5.1f}%)  "
          f"{ns_day:5.0f} ns/day  "
          f"ETA {eta_h:.1f} h", flush=True)

    # Roll to new trajectory file when this one exceeds ~5 GB (~87 ns)
    if (steps_done - STEPS_ALREADY) % (21_750_000) == 0 and steps_done < TOTAL_STEPS:
        seg_num  += 1
        traj_file = WORK / f"traj_{seg_num:03d}.dcd"
        log_file  = WORK / f"prod_{seg_num:03d}.log"
        sim.reporters.clear()
        sim.reporters.append(app.DCDReporter(str(traj_file), TRAJ_EVERY))
        sim.reporters.append(app.StateDataReporter(
            str(log_file), LOG_EVERY,
            step=True, time=True, temperature=True,
            density=True, potentialEnergy=True, speed=True))
        print(f"\n  >>> New segment: {traj_file.name}\n")

print(f"\n  ✓  Lurasidone — 500 ns COMPLETE!")
print(f"  Trajectories: {sorted(WORK.glob('traj_*.dcd'))}\n")
