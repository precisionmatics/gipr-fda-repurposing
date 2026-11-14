# GIPR FDA Drug Repurposing — Computational Pipeline

## Overview

End-to-end computational pipeline screening 1,123 FDA-approved compounds (Enamine library) against GIPR (PDB: 8WA3, peptide-free Gα_s-coupled active state). Four consensus hits — Lurasidone, Montelukast, Risperidone, and Atovaquone — were identified and characterised through seven independent metrics.

**Pipeline stages:**

| Stage | Method | Tool |
|-------|--------|------|
| 1 | Structural binding affinity prediction | Boltz-2 |
| 2 | Flexible docking | AutoDock-GPU |
| 3 | 100 ns MD stability | OpenMM + AMBER ff14SB/GAFF2 |
| 4 | Steered MD (mechanical unbinding) | OpenMM |
| 5 | MM-PBSA binding free energy | AmberTools MMPBSA.py |
| 6 | 500 ns extended MD (top 2) | OpenMM |

**Top compound: Lurasidone** (ΔG_bind = −13.54 ± 0.09 kcal mol⁻¹ at 500 ns; SMD rupture force 347 ± 9 pN; best combined multi-metric rank)

---

## Repository Structure

```
.
├── 01_virtual_screening/
│   ├── boltz_fda.py              # Boltz-2 IC50 prediction for all 1,123 compounds
│   ├── boltz_fda_resume.py       # Resume interrupted Boltz-2 run
│   └── check_and_collect.sh      # Collect Boltz-2 outputs
│
├── 02_docking/
│   └── dock_top30.py             # AutoDock-GPU docking analysis for top-30 candidates
│
├── 03_molecular_dynamics/
│   ├── run_md.py                 # 100 ns MD production run (OpenMM, all 4 hits)
│   ├── md_analysis.py            # RMSD, Rg, H-bond analysis of 100 ns trajectories
│   ├── run_all.sh                # Batch launcher for all 4 compounds
│   ├── run_montelukast.sh        # Montelukast-specific run script
│   ├── pose_quality.py           # Docking pose quality assessment
│   ├── quick_rmsd.py             # Rapid RMSD check
│   ├── extend_500ns/
│   │   ├── extend_md_500ns.py              # Extend Montelukast to 500 ns
│   │   └── extend_md_lurasidone_500ns.py   # Extend Lurasidone to 500 ns
│   └── leap_inputs/
│       └── {Compound}/
│           ├── tleap.in          # LEaP topology/solvation input
│           └── sqm.in            # AM1-BCC charge calculation input
│
├── 04_smd/
│   ├── smd_lurasidone.py         # SMD simulation — Lurasidone (5 replicates × 5 ns)
│   ├── smd_montelukast.py        # SMD simulation — Montelukast
│   ├── smd_risperidone.py        # SMD simulation — Risperidone
│   ├── smd_atovaquone.py         # SMD simulation — Atovaquone
│   ├── smd_analysis.py           # Rupture force extraction and statistics
│   └── smd_compare_all4.py       # Cross-compound comparison plots
│
├── 05_mmpbsa/
│   ├── run_mmgbsa_all4.py              # MM-PBSA for all 4 compounds (100 ns)
│   ├── run_mmpbsa_lurasidone_500ns.py  # MM-PBSA Lurasidone 500 ns
│   ├── run_mmgbsa_montelukast_500ns.py # MM-PBSA Montelukast 500 ns
│   └── input_files/
│       ├── 100ns/{Compound}/mmpbsa.in  # PB input (900 frames, istrng=150, inp=2)
│       └── 500ns/{Compound}/mmpbsa.in  # PB input (4900 frames)
│
├── 06_analysis_500ns/
│   └── analyze_500ns.py          # MDTraj 500 ns analysis (RMSD, Rg, H-bonds, figures)
│
└── results/
    └── smd_all4_analysis_results.json  # SMD rupture forces (mean ± SD, per replicate)
```

---

## Key Results

### MM-PBSA Binding Free Energies (ΔG_bind, kcal mol⁻¹)

| Compound    | 100 ns          | 500 ns          |
|-------------|-----------------|-----------------|
| Lurasidone  | −10.95 ± 0.18   | **−13.54 ± 0.09** |
| Atovaquone  | −5.24 ± 0.10    | —               |
| Montelukast | −1.62 ± 0.14    | −3.69 ± 0.07    |
| Risperidone | +3.80 ± 0.24    | —               |

### SMD Rupture Forces (pN)

| Compound    | Mean ± SD   |
|-------------|-------------|
| Lurasidone  | 347 ± 9     |
| Montelukast | 343 ± 33    |
| Risperidone | 349 ± 36    |
| Atovaquone  | 353 ± 43    |

---

## Dependencies

| Software | Version | Purpose |
|----------|---------|---------|
| OpenMM | 8.0 | MD / SMD simulations |
| AMBER / AmberTools | 22 | LEaP, AM1-BCC, MMPBSA.py |
| MDTraj | 1.9.9 | Trajectory analysis |
| Boltz-2 | latest | Structural affinity prediction |
| AutoDock-GPU | 1.5.3 | Docking |
| RDKit | 2024.03 | Ligand preparation |
| Python | 3.10+ | All scripts |

A conda environment file (`environment.yml`) with pinned versions is provided in the supplementary data deposit (Zenodo).

---

## Data Availability

Trajectory files, topology files, and supplementary data are deposited at Zenodo (DOI: *to be added upon publication*).

Supplementary screening tables (Tables S1–S5) accompany the manuscript.

---

## License

MIT License — see `LICENSE` file.
