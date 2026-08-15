#!/bin/bash
#SBATCH --job-name=train_all_grammars
#SBATCH --partition=GPU-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-191

# 192 array jobs, one per grammar; each job runs all 10 splits sequentially.

# assumes conda is on PATH
conda activate "$ENV"   # a Python env with the deps in requirements.txt
cd "$(dirname "$0")"
mkdir -p logs

SCRIPT="$(pwd)/train_gpt2.py"

core_types=("SVO" "SOV" "VSO" "VOS" "OSV" "OVS")
bools=("False" "True")

grammars=()
for core in "${core_types[@]}"; do
  for s2 in "${bools[@]}"; do
    for s3 in "${bools[@]}"; do
      for s4 in "${bools[@]}"; do
        for s5 in "${bools[@]}"; do
          for s6 in "${bools[@]}"; do
            grammars+=("${core}_s2${s2}_s3${s3}_s4${s4}_s5${s5}_s6${s6}")
          done
        done
      done
    done
  done
done

GRAMMAR="${grammars[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "BPE all-10-splits training — array task ${SLURM_ARRAY_TASK_ID}/191"
echo "Grammar : ${GRAMMAR}"
echo "Time    : $(date)"
echo "=========================================="

FAILED_SPLITS=()

for SPLIT in $(seq 0 9); do
  echo ""
  echo "--- Split ${SPLIT} ---"

  python "$SCRIPT" \
    --grammar "$GRAMMAR" \
    --split "$SPLIT"

  if [ $? -eq 0 ]; then
    echo "✓ Done: ${GRAMMAR}  split=${SPLIT}"
  else
    echo "✗ FAILED: ${GRAMMAR}  split=${SPLIT}"
    FAILED_SPLITS+=("$SPLIT")
  fi
done

echo ""
echo "=========================================="
echo "Finished all splits for ${GRAMMAR}"
echo "Time: $(date)"

if [ ${#FAILED_SPLITS[@]} -eq 0 ]; then
  echo "All 10 splits succeeded."
  exit 0
else
  echo "Failed splits: ${FAILED_SPLITS[*]}"
  exit 1
fi
