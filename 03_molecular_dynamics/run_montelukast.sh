#!/bin/bash
# Run Montelukast separately after charge fix (neutral form, charge=0)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Wait for current run_all.sh to finish (Risperidone + Atovaquone)
while pgrep -f "run_md.py Risperidone" > /dev/null || pgrep -f "run_md.py Atovaquone" > /dev/null; do
    sleep 60
done

echo "======================================================"
echo "  Montelukast (charge=0 neutral form) — $(date)"
echo "======================================================"

# Clean failed setup
rm -rf "$SCRIPT_DIR/Montelukast"

conda run -n unomd python "$SCRIPT_DIR/run_md.py" Montelukast >> "$SCRIPT_DIR/run_montelukast.log" 2>&1
if [ $? -eq 0 ]; then
    echo ">>> Montelukast DONE [$(date)]" | tee -a "$SCRIPT_DIR/run_montelukast.log"
else
    echo ">>> Montelukast FAILED [$(date)]" | tee -a "$SCRIPT_DIR/run_montelukast.log"
fi
