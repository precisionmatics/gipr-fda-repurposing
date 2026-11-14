"""
Boltz-2 binding affinity prediction for 1123 FDA-approved drugs vs GIPR.
Outputs:
  boltz/fda_boltz_results.tsv  — all compounds, every metric, sorted by IC50
  boltz/fda_top50.tsv          — top 50 by IC50
  boltz/top50_complexes/       — PDB protein-ligand structures for top 50
  boltz/top20_2D/              — 2D structure images for top 20
"""

import os, sys, json, shutil, subprocess, time, csv
from pathlib import Path
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path("/home/stalin/Downloads/Enamine/GIPR/FDA")
SMILES_CSV  = BASE / "Enamine_FDA_approved_Drugs_plated_1123cmpds_20250601.csv"
BOLTZ_DIR   = BASE / "boltz"
INPUTS_DIR  = BOLTZ_DIR / "inputs"
OUTPUTS_DIR = BOLTZ_DIR / "outputs"
MSA_CSV     = BOLTZ_DIR / "gipr_msa.csv"
RESULTS_TSV = BOLTZ_DIR / "fda_boltz_results.tsv"
TOP50_TSV   = BOLTZ_DIR / "fda_top50.tsv"
TOP50_DIR   = BOLTZ_DIR / "top50_complexes"
TOP20_DIR   = BOLTZ_DIR / "top20_2D"

for d in [INPUTS_DIR, OUTPUTS_DIR, TOP50_DIR, TOP20_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GIPR_SEQ = ("LDQRLILERLQVMYTVGYSLSLATLLLALLILSLFRRLHCTRNYIHINLFTSFMLRAAAILSRDRLLP"
             "NQALAACRTAQIVTQYCVGANYTWLLVEGVYLHSLLVLVGGSEEGHFRYYLLLGWGAPALFVIPWVI"
             "VRYLYENTQCWERNEVKAIWWIIRTPILMTILINFLIFIRILGILLSKLRTRQMRCRDYRLRLARSTL"
             "FLVPLLGVHEVVFAPVTEEQARGALRFAKLGFEIFLSSFQGFLVSVLYCFINKEVQSEIRRGWHHCR"
             "LRRS")

# ── 1. Write shared GIPR MSA CSV ──────────────────────────────────────────────
if not MSA_CSV.exists():
    MSA_CSV.write_text("key,sequence\n-1," + GIPR_SEQ + "\n")
    print(f"Wrote {MSA_CSV}")
else:
    print(f"MSA CSV already exists: {MSA_CSV}")

# ── 2. Load FDA compounds ──────────────────────────────────────────────────────
compounds = []
with open(SMILES_CSV) as f:
    first = f.readline()          # skip "sep=," line
    reader = csv.DictReader(f)
    for row in reader:
        compounds.append({
            "smiles_raw": row["SMILES"],
            "catalog_id": row["Catalog ID"],
            "MW":         row["MW"],
            "MW_desalted":row["MW (desalted)"],
            "ClogP":      row["ClogP"],
            "logS":       row["logS"],
            "HBD":        row["HBD"],
            "HBA":        row["HBA"],
            "TPSA":       row["TPSA"],
            "RotBonds":   row["RotBonds"],
        })

print(f"Loaded {len(compounds)} FDA compounds")

# ── 3. Clean SMILES with RDKit (desalt, strip CXSmiles) ───────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.MolStandardize import rdMolStandardize
    HAS_RDKIT = True
    print("RDKit available — will desalt SMILES")
except ImportError:
    HAS_RDKIT = False
    print("WARNING: RDKit not found — using raw SMILES (salts not removed)")

def clean_smiles(smi):
    """Strip CXSmiles annotations and desalt using RDKit."""
    smi = smi.split(" |")[0].strip()
    if not HAS_RDKIT:
        return smi
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        # pick largest fragment (desalt)
        frags = rdMolStandardize.TautomerEnumerator().Canonicalize
        chooser = rdMolStandardize.LargestFragmentChooser()
        mol = chooser.choose(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return smi

skipped = []
for c in compounds:
    c["smiles_clean"] = clean_smiles(c["smiles_raw"])
    if not c["smiles_clean"]:
        skipped.append(c["catalog_id"])

if skipped:
    print(f"WARNING: {len(skipped)} compounds with empty cleaned SMILES: {skipped[:5]}")

# ── 4. Generate YAML inputs ────────────────────────────────────────────────────
yaml_files = []
for c in compounds:
    cid = c["catalog_id"].replace("/", "_").replace(" ", "_")
    yaml_path = INPUTS_DIR / f"{cid}.yaml"
    yaml_files.append((yaml_path, c))
    if yaml_path.exists():
        continue
    smi = c["smiles_clean"] or c["smiles_raw"]
    yaml_path.write_text(
        f"version: 1\n"
        f"sequences:\n"
        f"  - protein:\n"
        f"      id: A\n"
        f"      sequence: {GIPR_SEQ}\n"
        f"      msa: {MSA_CSV}\n"
        f"  - ligand:\n"
        f"      id: B\n"
        f"      smiles: '{smi}'\n"
        f"properties:\n"
        f"  - affinity:\n"
        f"      binder: B\n"
    )

already_done = 0
PRED_ROOT = OUTPUTS_DIR / "boltz_results_inputs" / "predictions"
if PRED_ROOT.exists():
    for d in PRED_ROOT.iterdir():
        if d.is_dir() and any(d.glob("affinity_*.json")):
            already_done += 1

print(f"YAML files ready: {len(yaml_files)}  |  Already predicted: {already_done}  |  Remaining: {len(yaml_files) - already_done}")

# ── 5. Run Boltz-2 ────────────────────────────────────────────────────────────
cmd = [
    "conda", "run", "-n", "boltz", "--no-capture-output",
    "boltz", "predict", str(INPUTS_DIR),
    "--out_dir", str(OUTPUTS_DIR),
    "--accelerator", "gpu",
    "--devices", "1",
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

print(f"\nRunning Boltz-2 on {len(yaml_files)} FDA compounds...")
print(f"CMD: {' '.join(cmd)}\n")
t0 = time.time()
result = subprocess.run(cmd, cwd=str(BASE))
elapsed = time.time() - t0
print(f"\nBoltz-2 finished in {elapsed/3600:.1f}h  (exit code {result.returncode})")

# ── 6. Parse all outputs ───────────────────────────────────────────────────────
print("\nParsing results...")

# Boltz output dir: OUTPUTS_DIR/<inputs_dir_stem>/predictions/<compound>/
# inputs_dir_stem = "inputs" → folder = "boltz_fda_inputs" (boltz appends to dir name)
PRED_ROOT = OUTPUTS_DIR / "boltz_results_inputs" / "predictions"
if not PRED_ROOT.exists():
    # fallback: scan all prediction dirs
    candidates = [d for d in OUTPUTS_DIR.rglob("predictions") if d.is_dir()]
    if candidates:
        PRED_ROOT = candidates[0]
        print(f"Using predictions dir: {PRED_ROOT}")
    else:
        print("ERROR: No predictions directory found!")
        sys.exit(1)

# Build catalog_id lookup keyed by YAML stem (= catalog_id)
info_by_id = {c["catalog_id"].replace("/", "_").replace(" ", "_"): c for c in compounds}

records = []
missing = []

for pred_dir in sorted(PRED_ROOT.iterdir()):
    if not pred_dir.is_dir():
        continue
    stem = pred_dir.name

    # Parse affinity JSON
    aff_files = list(pred_dir.glob("affinity_*.json"))
    if not aff_files:
        missing.append(stem)
        continue
    try:
        aff_data = json.loads(aff_files[0].read_text())
    except Exception as e:
        print(f"  ERROR reading affinity JSON for {stem}: {e}")
        continue

    affinity_main = aff_data.get("affinity_pred_value")
    if affinity_main is None:
        print(f"  WARNING: no affinity_pred_value in {aff_files[0]}")
        continue

    # Collect all sample values
    sample_affinities = [float(affinity_main)]
    sample_probs      = [float(aff_data.get("affinity_probability_binary", float("nan")))]
    for i in range(1, 10):
        key_a = f"affinity_pred_value{i}"
        key_p = f"affinity_probability_binary{i}"
        if key_a in aff_data:
            sample_affinities.append(float(aff_data[key_a]))
            sample_probs.append(float(aff_data.get(key_p, float("nan"))))
        else:
            break

    # Parse confidence/structural quality JSON
    conf_data = {}
    conf_files = list(pred_dir.glob("confidence_*.json"))
    if conf_files:
        try:
            conf_data = json.loads(conf_files[0].read_text())
        except Exception:
            pass

    # Compute IC50: affinity_pred_value = log10(IC50 in μM) → lower = stronger
    ic50_nM       = 10 ** float(affinity_main) * 1000.0
    ic50_nM_mean  = 10 ** np.mean(sample_affinities) * 1000.0
    ic50_nM_best  = 10 ** min(sample_affinities) * 1000.0

    c = info_by_id.get(stem, {})
    records.append({
        "stem":                    stem,
        "catalog_id":              c.get("catalog_id", stem),
        "smiles_original":         c.get("smiles_raw", ""),
        "smiles_clean":            c.get("smiles_clean", ""),
        "MW":                      c.get("MW", ""),
        "MW_desalted":             c.get("MW_desalted", ""),
        "ClogP":                   c.get("ClogP", ""),
        "logS":                    c.get("logS", ""),
        "HBD":                     c.get("HBD", ""),
        "HBA":                     c.get("HBA", ""),
        "TPSA":                    c.get("TPSA", ""),
        "RotBonds":                c.get("RotBonds", ""),
        # Affinity metrics
        "affinity_pred_value":     float(affinity_main),
        "affinity_probability_binary": sample_probs[0],
        "n_samples":               len(sample_affinities),
        "affinity_mean":           float(np.mean(sample_affinities)),
        "affinity_std":            float(np.std(sample_affinities)),
        "affinity_best":           float(min(sample_affinities)),
        "prob_binary_mean":        float(np.nanmean(sample_probs)),
        "IC50_nM":                 ic50_nM,
        "IC50_nM_mean":            ic50_nM_mean,
        "IC50_nM_best":            ic50_nM_best,
        # All sample values (for TSV)
        "sample_affinities":       sample_affinities,
        "sample_probs":            sample_probs,
        # Structural quality
        "confidence_score":        conf_data.get("confidence_score", float("nan")),
        "ptm":                     conf_data.get("ptm", float("nan")),
        "iptm":                    conf_data.get("iptm", float("nan")),
        "ligand_iptm":             conf_data.get("ligand_iptm", float("nan")),
        "complex_plddt":           conf_data.get("complex_plddt", float("nan")),
        "complex_iplddt":          conf_data.get("complex_iplddt", float("nan")),
        "complex_pde":             conf_data.get("complex_pde", float("nan")),
        # PDB path
        "pdb_path":                next(pred_dir.glob("*_model_0.pdb"), None),
    })

if missing:
    print(f"  Missing affinity results for {len(missing)} compounds")

# Sort by IC50 ascending (lower = stronger binder)
records.sort(key=lambda x: x["IC50_nM"])

print(f"\nParsed {len(records)} compounds with affinity predictions")

# ── 7. Write full results TSV ─────────────────────────────────────────────────
n_samples = max(r["n_samples"] for r in records) if records else 1
sample_aff_cols  = [f"affinity_pred_value{i}" for i in range(n_samples)]
sample_prob_cols = [f"prob_binary{i}" for i in range(n_samples)]

header = (
    ["rank", "catalog_id", "IC50_nM", "IC50_nM_mean", "IC50_nM_best",
     "affinity_pred_value", "affinity_mean", "affinity_std", "affinity_best",
     "affinity_probability_binary", "prob_binary_mean"]
    + sample_aff_cols + sample_prob_cols
    + ["confidence_score", "ptm", "iptm", "ligand_iptm",
       "complex_plddt", "complex_iplddt", "complex_pde",
       "MW", "MW_desalted", "ClogP", "logS", "HBD", "HBA", "TPSA", "RotBonds",
       "smiles_clean", "smiles_original"]
)

with open(RESULTS_TSV, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(header)
    for rank, r in enumerate(records, 1):
        sa = r["sample_affinities"]
        sp = r["sample_probs"]
        aff_vals  = [f"{v:.4f}" for v in sa] + [""] * (n_samples - len(sa))
        prob_vals = [f"{v:.4f}" for v in sp] + [""] * (n_samples - len(sp))
        row = (
            [rank, r["catalog_id"],
             f"{r['IC50_nM']:.1f}", f"{r['IC50_nM_mean']:.1f}", f"{r['IC50_nM_best']:.1f}",
             f"{r['affinity_pred_value']:.4f}", f"{r['affinity_mean']:.4f}",
             f"{r['affinity_std']:.4f}", f"{r['affinity_best']:.4f}",
             f"{r['affinity_probability_binary']:.4f}", f"{r['prob_binary_mean']:.4f}"]
            + aff_vals + prob_vals
            + [f"{r['confidence_score']:.4f}" if r['confidence_score'] == r['confidence_score'] else "",
               f"{r['ptm']:.4f}" if r['ptm'] == r['ptm'] else "",
               f"{r['iptm']:.4f}" if r['iptm'] == r['iptm'] else "",
               f"{r['ligand_iptm']:.4f}" if r['ligand_iptm'] == r['ligand_iptm'] else "",
               f"{r['complex_plddt']:.4f}" if r['complex_plddt'] == r['complex_plddt'] else "",
               f"{r['complex_iplddt']:.4f}" if r['complex_iplddt'] == r['complex_iplddt'] else "",
               f"{r['complex_pde']:.4f}" if r['complex_pde'] == r['complex_pde'] else "",
               r["MW"], r["MW_desalted"], r["ClogP"], r["logS"],
               r["HBD"], r["HBA"], r["TPSA"], r["RotBonds"],
               r["smiles_clean"], r["smiles_original"]]
        )
        w.writerow(row)

print(f"Full results → {RESULTS_TSV}  ({len(records)} compounds)")

# ── 8. Write top-50 TSV ───────────────────────────────────────────────────────
top50 = records[:50]
shutil.copy(RESULTS_TSV, TOP50_TSV.with_suffix(".tmp"))
with open(TOP50_TSV, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(header)
    for rank, r in enumerate(top50, 1):
        sa = r["sample_affinities"]
        sp = r["sample_probs"]
        aff_vals  = [f"{v:.4f}" for v in sa] + [""] * (n_samples - len(sa))
        prob_vals = [f"{v:.4f}" for v in sp] + [""] * (n_samples - len(sp))
        row = (
            [rank, r["catalog_id"],
             f"{r['IC50_nM']:.1f}", f"{r['IC50_nM_mean']:.1f}", f"{r['IC50_nM_best']:.1f}",
             f"{r['affinity_pred_value']:.4f}", f"{r['affinity_mean']:.4f}",
             f"{r['affinity_std']:.4f}", f"{r['affinity_best']:.4f}",
             f"{r['affinity_probability_binary']:.4f}", f"{r['prob_binary_mean']:.4f}"]
            + aff_vals + prob_vals
            + [f"{r['confidence_score']:.4f}" if r['confidence_score'] == r['confidence_score'] else "",
               f"{r['ptm']:.4f}" if r['ptm'] == r['ptm'] else "",
               f"{r['iptm']:.4f}" if r['iptm'] == r['iptm'] else "",
               f"{r['ligand_iptm']:.4f}" if r['ligand_iptm'] == r['ligand_iptm'] else "",
               f"{r['complex_plddt']:.4f}" if r['complex_plddt'] == r['complex_plddt'] else "",
               f"{r['complex_iplddt']:.4f}" if r['complex_iplddt'] == r['complex_iplddt'] else "",
               f"{r['complex_pde']:.4f}" if r['complex_pde'] == r['complex_pde'] else "",
               r["MW"], r["MW_desalted"], r["ClogP"], r["logS"],
               r["HBD"], r["HBA"], r["TPSA"], r["RotBonds"],
               r["smiles_clean"], r["smiles_original"]]
        )
        w.writerow(row)
TOP50_TSV.with_suffix(".tmp").unlink(missing_ok=True)
print(f"Top-50  → {TOP50_TSV}")

# ── 9. Copy top-50 PDB complexes ──────────────────────────────────────────────
print(f"\nCopying top-50 PDB complexes → {TOP50_DIR}")
copied = 0
for rank, r in enumerate(top50, 1):
    pdb = r["pdb_path"]
    if pdb and Path(pdb).exists():
        dest = TOP50_DIR / f"rank{rank:02d}_{r['catalog_id']}_IC50_{r['IC50_nM']:.0f}nM.pdb"
        shutil.copy(pdb, dest)
        copied += 1
    else:
        print(f"  WARNING: no PDB for rank {rank} ({r['catalog_id']})")
print(f"  Copied {copied} PDB files")

# ── 10. Generate 2D structure images for top-20 ───────────────────────────────
if HAS_RDKIT:
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    print(f"\nGenerating 2D images for top-20 → {TOP20_DIR}")
    for rank, r in enumerate(records[:20], 1):
        smi = r["smiles_clean"] or r["smiles_original"]
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            AllChem.Compute2DCoords(mol)
            drawer = rdMolDraw2D.MolDraw2DSVG(600, 400)
            drawer.drawOptions().addStereoAnnotation = True
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            out = TOP20_DIR / f"rank{rank:02d}_{r['catalog_id']}_IC50_{r['IC50_nM']:.0f}nM.svg"
            out.write_text(svg)
        except Exception as e:
            print(f"  2D image failed for rank {rank}: {e}")
    print(f"  Done")
else:
    print("Skipping 2D images (RDKit not available)")

# ── 11. Print summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"  GIPR × FDA DRUGS — BOLTZ-2 RESULTS SUMMARY")
print("=" * 80)
print(f"  Total predicted: {len(records)} / {len(compounds)} compounds")

ic50s = [r["IC50_nM"] for r in records]
sub100  = sum(1 for x in ic50s if x < 100)
sub1000 = sum(1 for x in ic50s if x < 1000)
sub10um = sum(1 for x in ic50s if x < 10000)
print(f"  Sub-100 nM:   {sub100}")
print(f"  Sub-1 μM:     {sub1000}")
print(f"  Sub-10 μM:    {sub10um}")
print()
print(f"  {'Rank':<5} {'IC50(nM)':>10} {'IC50_mean':>10} {'Prob':>6} {'Conf':>6}  Catalog ID")
print(f"  {'-'*4:<5} {'-'*9:>10} {'-'*9:>10} {'-'*5:>6} {'-'*5:>6}  {'-'*20}")
for rank, r in enumerate(records[:20], 1):
    prob = r["affinity_probability_binary"]
    conf = r["confidence_score"]
    prob_s = f"{prob:.2f}" if prob == prob else "N/A"
    conf_s = f"{conf:.3f}" if conf == conf else "N/A"
    print(f"  {rank:<5} {r['IC50_nM']:>10.1f} {r['IC50_nM_mean']:>10.1f} {prob_s:>6} {conf_s:>6}  {r['catalog_id']}")
print("=" * 80)
print(f"\nOutput files:")
print(f"  {RESULTS_TSV}")
print(f"  {TOP50_TSV}")
print(f"  {TOP50_DIR}/  ({copied} PDB files)")
print(f"  {TOP20_DIR}/  (2D images)")
