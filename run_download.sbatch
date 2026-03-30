#!/bin/bash
#SBATCH --job-name=download_ngrams
#SBATCH --output=logs/download_ngrams_%j.out
#SBATCH --error=logs/download_ngrams_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=gpu

# ============================================================================
# DOWNLOAD PARALLELO GOOGLE BOOKS N-GRAMS
# ============================================================================
# Questo script scarica 130 file (5 per ogni lettera A-Z) in parallelo
# usando 8 worker simultanei. Stima spazio: ~45 GB
#
# Uso:
#   sbatch run_download.sbatch
#
# Monitoraggio:
#   squeue -u $USER
#   tail -f logs/download_ngrams_<job_id>.out
# ============================================================================

echo "=============================================================================="
echo "JOB: Download Google Books N-grams (parallelo)"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 16GB"
echo "Start time: $(date)"
echo "=============================================================================="
echo ""

# Attiva ambiente virtuale se esiste
if [ -d ".venv" ]; then
    echo "Attivando ambiente virtuale..."
    source .venv/bin/activate
fi

# Verifica Python e moduli necessari
echo "Python: $(which python3)"
echo "Version: $(python3 --version)"
echo ""

# Verifica spazio disco disponibile
echo "Spazio disco disponibile:"
df -h $HOME | tail -1
echo ""

# Crea directory logs se non esiste
mkdir -p logs

# Esegui download parallelo
echo "Avvio download parallelo con 8 worker..."
echo "Output: data/raw/5gram-expanded/5gram_filtered_expanded.tsv"
echo ""

python3 src/data/download_parallel.py

EXIT_CODE=$?

echo ""
echo "=============================================================================="
echo "Download completato!"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=============================================================================="
echo ""

# Statistiche finali
if [ -d "data/raw/5gram-expanded" ]; then
    echo "Statistiche finali:"
    echo "  File scaricati: $(ls -1 data/raw/5gram-expanded/*.gz 2>/dev/null | wc -l)"
    echo "  Spazio utilizzato: $(du -sh data/raw/5gram-expanded/ 2>/dev/null | cut -f1)"
    
    if [ -f "data/raw/5gram-expanded/5gram_filtered_expanded.tsv" ]; then
        echo "  File output: $(du -sh data/raw/5gram-expanded/5gram_filtered_expanded.tsv | cut -f1)"
        echo "  Righe output: $(wc -l < data/raw/5gram-expanded/5gram_filtered_expanded.tsv)"
    fi
fi

exit $EXIT_CODE
