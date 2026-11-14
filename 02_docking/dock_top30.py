"""
Dock top 30 FDA drug-like hits vs GIPR (8WA3).
Uses same autodock_gpu_ocl + receptor maps as the Enamine MEMES pipeline.
"""

import subprocess, sys, re, shutil
from pathlib import Path

BASE         = Path("/home/stalin/Downloads/Enamine/GIPR")
RECEPTOR     = BASE / "gipr_8WA3.maps.fld"
AUTODOCK_BIN = BASE / "autodock_gpu_ocl"
SMILES_FILE  = BASE / "FDA/results/top30_for_docking.smi"
WORK         = BASE / "FDA/results/docking"
PDBQT_DIR    = WORK / "pdbqt"
DOCK_DIR     = WORK / "dock_results"

WORK.mkdir(parents=True, exist_ok=True)
PDBQT_DIR.mkdir(exist_ok=True)
DOCK_DIR.mkdir(exist_ok=True)

NRUN = 20   # more runs than screening (20) for better accuracy on 30 compounds

# ── 1. Load SMILES ─────────────────────────────────────────────────────────────
compounds = []
with open(SMILES_FILE) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            compounds.append((parts[1], parts[0]))  # (cid, smiles)
print(f"Loaded {len(compounds)} compounds")

# ── 2. Prepare PDBQT files ────────────────────────────────────────────────────
try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_MEEKO = True
except ImportError:
    HAS_MEEKO = False
    print("ERROR: meeko not found")
    sys.exit(1)

preparator = MoleculePreparation()
prepped = []
failed = []

for cid, smi in compounds:
    out = PDBQT_DIR / f"{cid}.pdbqt"
    if out.exists():
        prepped.append(cid)
        continue
    try:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        setups = preparator.prepare(mol)
        pdbqt_str, _, _ = PDBQTWriterLegacy.write_string(setups[0])
        out.write_text(pdbqt_str)
        prepped.append(cid)
    except Exception as e:
        failed.append((cid, str(e)))
        print(f"  FAIL prep {cid}: {e}")

print(f"PDBQT ready: {len(prepped)}  |  Failed: {len(failed)}")

# ── 3. Run AutoDock-GPU (OCL) ─────────────────────────────────────────────────
filelist = DOCK_DIR / "filelist.txt"
with open(filelist, "w") as f:
    f.write(str(RECEPTOR) + "\n")
    for cid in prepped:
        f.write(str(PDBQT_DIR / f"{cid}.pdbqt") + "\n")

cmd = [str(AUTODOCK_BIN), "--filelist", str(filelist),
       "--nrun", str(NRUN), "--xmloutput", "1"]
print(f"\nRunning AutoDock-GPU on {len(prepped)} compounds (nrun={NRUN})...")
print(f"CMD: {' '.join(cmd)}\n")
subprocess.run(cmd, cwd=str(DOCK_DIR))

# ── 4. Parse XML scores ────────────────────────────────────────────────────────
def parse_best_score(xml_path):
    text = xml_path.read_text()
    scores = re.findall(r'<run[^>]*>\s*<free_NRG_binding>([-\d.]+)', text)
    if not scores:
        # fallback pattern
        scores = re.findall(r'<free_NRG_binding>([-\d.]+)', text)
    return min(float(s) for s in scores) if scores else None

print("\nParsing docking results...")
scores = {}
for cid in prepped:
    xml = DOCK_DIR / f"{cid}.xml"
    if xml.exists():
        s = parse_best_score(xml)
        if s is not None:
            scores[cid] = s
    else:
        # try alternative naming
        for xml in DOCK_DIR.glob(f"*{cid}*.xml"):
            s = parse_best_score(xml)
            if s is not None:
                scores[cid] = s
            break

print(f"Scores parsed: {len(scores)} / {len(prepped)}")

# ── 5. Load Boltz-2 results and merge ────────────────────────────────────────
import json, csv
from pathlib import Path

PRED_ROOT  = Path("/home/stalin/Downloads/Enamine/GIPR/FDA/boltz/outputs/boltz_results_inputs/predictions")
SMILES_CSV = Path("/home/stalin/Downloads/Enamine/GIPR/FDA/Enamine_FDA_approved_Drugs_plated_1123cmpds_20250601.csv")

info = {}
with open(SMILES_CSV) as f:
    f.readline()
    for row in csv.DictReader(f):
        info[row['Catalog ID']] = row

boltz = {}
for pred_dir in PRED_ROOT.iterdir():
    aff_files = list(pred_dir.glob('affinity_*.json'))
    if not aff_files: continue
    try:
        d = json.loads(aff_files[0].read_text())
        aff = float(d['affinity_pred_value'])
        boltz[pred_dir.name] = {
            'ic50_nM': 10**aff * 1000,
            'affinity': aff,
            'prob': float(d.get('affinity_probability_binary', 0))
        }
    except: pass

# Build final table
rows = []
for cid, smiles in compounds:
    c = info.get(cid, {})
    b = boltz.get(cid, {})
    rows.append({
        'catalog_id':    cid,
        'dock_score':    f"{scores.get(cid, ''):.3f}" if cid in scores else 'N/A',
        'IC50_nM':       f"{b.get('ic50_nM', ''):.1f}" if b else 'N/A',
        'affinity_pred': f"{b.get('affinity', ''):.4f}" if b else 'N/A',
        'prob_binary':   f"{b.get('prob', ''):.3f}" if b else 'N/A',
        'MW':            c.get('MW (desalted)', ''),
        'ClogP':         c.get('ClogP', ''),
        'HBD':           c.get('HBD', ''),
        'HBA':           c.get('HBA', ''),
        'TPSA':          c.get('TPSA', ''),
        'smiles':        smiles,
    })

# Sort by docking score (lowest = best)
rows.sort(key=lambda x: float(x['dock_score']) if x['dock_score'] != 'N/A' else 99)

out_csv = BASE / "FDA/results/top30_docking_results.csv"
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\nResults → {out_csv}")
print(f"\n{'Rank':<5} {'Dock(kcal/mol)':>15} {'IC50_nM':>10} {'Prob':>6} {'MW':>7}  Catalog ID")
print('-'*70)
for rank, r in enumerate(rows, 1):
    print(f"{rank:<5} {r['dock_score']:>15} {r['IC50_nM']:>10} {r['prob_binary']:>6} {r['MW']:>7}  {r['catalog_id']}")
