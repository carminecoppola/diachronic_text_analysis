#!/bin/bash

#SBATCH --job-name=diachronic_cbow_training
#SBATCH --output=logs/training_%j.log
#SBATCH --error=logs/training_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu

# ============================================================================
# SBATCH per Training CBOW (FASE 5)
# ============================================================================
# Esecuzione:
#   sbatch run_training.sbatch
#
# Monitoraggio:
#   squeue -u $USER
#   tail -f logs/training_*.log
#
# Risorse richieste:
#   - 16 CPUs
#   - 64 GB RAM
#   - 4 GPUs (per distributed training)
#   - 24 ore max
# ============================================================================

# Carica moduli HPC (adatta a tuo cluster)
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

# Crea directory log se non esiste
mkdir -p logs
mkdir -p models
mkdir -p plots

# Attiva ambiente virtuale
cd /home/vbucciero/diachronic_text_analysis
source .venv/bin/activate

# Info job
echo "=========================================="
echo "🚀 SBATCH: CBOW Training (FASE 5)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Time: $(date)"
echo "=========================================="
echo ""

# Export PyTorch environment variables
export CUDA_VISIBLE_DEVICES=0
export TORCH_HOME=/scratch/$USER/torch
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH=/home/vbucciero/diachronic_text_analysis/src:$PYTHONPATH

# Log configurazione
echo "🔧 Configuration:"
echo "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  - OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  - TORCH_HOME: $TORCH_HOME"
echo ""

# Esegui training
echo "🚀 Starting training..."
python src/train.py

# Cattura exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ Training completato con successo"
else
  echo "❌ Training fallito (exit code: $EXIT_CODE)"
fi
echo "Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
