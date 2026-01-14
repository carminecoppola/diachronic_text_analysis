# Training Word Embeddings

## Architettura CBOW

Continuous Bag of Words con skip-gram negativo.

### Modello

File: `src/models/model.py`

Componenti:
- Input embedding layer: vocab_size × embedding_dim
- Output embedding layer: vocab_size × embedding_dim
- Negative sampling per efficienza

### Forward Pass

```
Context: [word1, word3]  (da n-gram "word1 word2 word3")
Target: word2

1. Embed context words
2. Average context embeddings
3. Compute similarity con target
4. Compute similarity con negative samples
5. Binary cross-entropy loss
```

### Negative Sampling

Numero samples: 3 (configurabile)
- Riduce complessità da O(V) a O(k)
- Sampling proporzionale a freq^0.75
- Evita parole troppo frequenti come negative

## Dataset PyTorch

File: `src/data/dataset.py`

Implementazione:
```python
class CBOWDataset(Dataset):
    def __getitem__(self, idx):
        # Legge n-gram da file
        # Estrae target e context
        # Ritorna tensor indices
```

Features:
- Lazy loading (non carica tutto in RAM)
- Sampling efficiente con mmap
- Support per 3-gram e 5-gram

## Training Loop

Script: `src/models/train.py`
SLURM: `run_training.sbatch`

### Parametri

```python
EMBEDDING_DIM = 300      # Standard Word2Vec
BATCH_SIZE = 512
LEARNING_RATE = 0.025    # Con decay
EPOCHS = 3
CONTEXT_WINDOW = 2       # Per 5-gram
NEGATIVE_SAMPLES = 3
```

### Processo

Per ogni decennio (1900s-2010s):
1. Load vocabolario comune
2. Create dataset da file processed
3. Train modello CBOW
4. Save embeddings

Output: `models/{3gram,5gram}/cbow_{decade}.pt`

Checkpoint structure:
```python
{
    'decade': '1990s',
    'embedding_dim': 300,
    'vocab_size': 50001,
    'input_embeddings': tensor(...),   # Usati per analisi
    'output_embeddings': tensor(...),
    'vocab': {...}
}
```

### SLURM Configuration

```bash
#SBATCH --job-name=cbow_train
#SBATCH --array=0-11           # 12 decenni
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
```

## Ottimizzazioni

### Memory

- Streaming dataset (no full load)
- Batch processing
- Gradient accumulation se necessario

### Speed

- DataLoader num_workers=8
- GPU acceleration
- Prefetching

### Quality

- Learning rate decay
- Early stopping su validation loss (se implementato)
- Multiple epochs per convergenza

## Validazione

Metriche durante training:
- Loss per epoch
- Convergenza gradient norm
- Sample predictions (nearest neighbors)

Post-training:
- Analogies (se implementato)
- Clustering quality
- Sanity check embedding norms

## Output

Modelli salvati in: `models/{3gram,5gram}/`

Naming: `cbow_1900s.pt`, `cbow_1910s.pt`, ..., `cbow_2010s.pt`

Size: ~600MB per modello (300dim × 50K vocab × 2 matrices)

Total: ~7.2GB per tipo n-gram
