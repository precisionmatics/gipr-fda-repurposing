"""
Resume/restart script for FDA Boltz-2 run.
Run this after a power cut or after the main run finishes.
It checks what's already done and re-runs only missing compounds in batches of 100.
"""

import subprocess, shutil, json
from pathlib import Path

BASE        = Path("/home/stalin/Downloads/Enamine/GIPR/FDA")
INPUTS_DIR  = BASE / "boltz/inputs"
OUTPUTS_DIR = BASE / "boltz/outputs"
PRED_ROOT   = OUTPUTS_DIR / "boltz_results_inputs" / "predictions"

BATCH_SIZE  = 100
BATCH_DIR   = BASE / "boltz/batch_inputs"

def get_done():
    done = set()
    if PRED_ROOT.exists():
        for d in PRED_ROOT.iterdir():
            if d.is_dir() and any(d.glob("affinity_*.json")):
                done.add(d.name)
    return done

all_yamls = sorted(INPUTS_DIR.glob("*.yaml"))
done = get_done()
remaining = [y for y in all_yamls if y.stem not in done]

print(f"Total YAMLs:     {len(all_yamls)}")
print(f"Already done:    {len(done)}")
print(f"Remaining:       {len(remaining)}")

if not remaining:
    print("All done! Run boltz_fda.py to collect results.")
else:
    BATCH_DIR.mkdir(exist_ok=True)
    # clear old batch inputs
    for f in BATCH_DIR.glob("*.yaml"):
        f.unlink()

    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start:batch_start + BATCH_SIZE]
        # clear and fill batch dir
        for f in BATCH_DIR.glob("*.yaml"):
            f.unlink()
        for y in batch:
            shutil.copy(y, BATCH_DIR / y.name)

        done_now = get_done()
        still_needed = [y for y in batch if y.stem not in done_now]
        if not still_needed:
            print(f"Batch {batch_start//BATCH_SIZE + 1}: already done, skipping")
            continue

        print(f"\nBatch {batch_start//BATCH_SIZE + 1}: running {len(still_needed)} compounds "
              f"({batch_start+1}–{batch_start+len(batch)} of {len(remaining)})...")

        cmd = [
            "conda", "run", "-n", "boltz", "--no-capture-output",
            "boltz", "predict", str(BATCH_DIR),
            "--out_dir", str(OUTPUTS_DIR),
            "--accelerator", "gpu", "--devices", "1",
            "--model", "boltz2",
            "--diffusion_samples", "1",
            "--diffusion_samples_affinity", "5",
            "--sampling_steps", "100",
            "--sampling_steps_affinity", "100",
            "--affinity_mw_correction",
            "--recycling_steps", "3",
            "--output_format", "pdb",
            "--max_parallel_samples", "1",
        ]
        result = subprocess.run(cmd, cwd=str(BASE))
        done_after = get_done()
        print(f"  Done so far: {len(done_after)} / {len(all_yamls)}")
        if result.returncode != 0:
            print(f"  WARNING: non-zero exit code {result.returncode}")

    print("\nAll batches complete. Run boltz_fda.py to collect and analyze results.")
