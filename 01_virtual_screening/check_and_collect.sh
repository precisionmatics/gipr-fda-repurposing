#!/bin/bash
# Wait for all affinity predictions to finish, then collect results
TARGET=1118
PRED_DIR="/home/stalin/Downloads/Enamine/GIPR/FDA/boltz/outputs/boltz_results_inputs/predictions"
LOG="/home/stalin/Downloads/Enamine/GIPR/FDA/boltz/collect.log"

echo "$(date): Watching for $TARGET affinity JSONs..." | tee $LOG

while true; do
    done=$(find $PRED_DIR -name "affinity_*.json" 2>/dev/null | wc -l)
    echo "$(date): $done / $TARGET done" | tee -a $LOG
    if [ "$done" -ge "$TARGET" ]; then
        echo "$(date): All done! Running results collection..." | tee -a $LOG
        cd /home/stalin/Downloads/Enamine/GIPR/FDA
        conda run -n boltz python3 boltz_fda.py >> $LOG 2>&1
        echo "$(date): COMPLETE. Check boltz/fda_boltz_results.tsv" | tee -a $LOG
        break
    fi
    sleep 300
done
