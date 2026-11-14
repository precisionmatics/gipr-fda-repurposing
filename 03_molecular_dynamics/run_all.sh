#!/bin/bash
# Run all 4 GIPR-ligand MD simulations sequentially (100 ns each).
# Fully resumable — re-run this script after any power loss to continue.
# Each compound picks up exactly where it left off.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/run_md.py"

echo "======================================================"
echo "  GIPR FDA MD — 4 compounds × 100 ns"
echo "  Started: $(date)"
echo "======================================================"

for compound in Lurasidone Montelukast Risperidone Atovaquone; do
    echo ""
    echo ">>> $compound  [$(date)]"
    conda run -n unomd python "$PYTHON_SCRIPT" "$compound"
    if [ $? -eq 0 ]; then
        echo ">>> $compound DONE [$(date)]"
    else
        echo ">>> $compound FAILED — check logs in production/$compound/"
        echo "    Continuing with next compound..."
    fi
done

echo ""
echo "======================================================"
echo "  All compounds finished: $(date)"
echo "======================================================"
