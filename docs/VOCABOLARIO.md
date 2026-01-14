# Vocabolario

## Definizione

Dizionario che mappa parole a indici numerici per uso nel modello CBOW.

Struttura:
```json
{
  "word2idx": {"<UNK>": 0, "the": 1, "of": 2, "computer": 1523},
  "idx2word": ["<UNK>", "the", "of", ..., "computer", ...],
  "word_freq": {"the": 123456789, "of": 98765432, ...}
}
```

## Funzioni

1. Conversione testo → numeri per modello
2. Vocabolario comune tra tutti decenni (necessario per confronto embeddings)
3. Gestione out-of-vocabulary con token UNK

## Creazione

Script: `src/data/build_vocab.py`

Processo:
1. Estrae parole singole da n-gram di tutti decenni
2. Calcola frequenze totali cross-decade
3. Ordina per frequenza decrescente
4. Seleziona top 50,000
5. Aggiunge UNK come indice 0

Output: `data/processed/{3gram,5gram}/vocab.json`

## Statistiche

- Dimensione: 50,001 parole (50K + UNK)
- Top words: "the", "of", "and", "to", "in"
- Copertura: ~99.98% del corpus
- Parole uniche totali nel corpus: ~15M

## Considerazioni

### Vocabolario Comune

Stesso vocabolario per tutti i decenni:
- Permette confronto diretto tra embeddings
- Parole nuove (post-1900) mappate a UNK
- Focus su parole presenti attraverso il secolo

### Threshold

Top 50K scelto per bilanciare:
- Copertura corpus (>99%)
- Dimensione modello gestibile
- Riduzione noise da parole rare

### Token UNK

Gestisce:
- Parole fuori vocabolario
- Typos residui dopo filtering
- Neologismi decade-specific
- Parole sotto threshold frequenza

## Utilizzo nel Training

Durante training CBOW:
```python
# Conversione parola → indice
word_idx = vocab['word2idx'].get(word, 0)  # 0 = UNK

# Conversione indice → embedding
embedding = embedding_layer(word_idx)
```

Durante inference:
```python
# Nearest neighbors
word_vec = embeddings[vocab['word2idx']['computer']]
similarities = cosine_similarity(word_vec, all_embeddings)
```

## Validazione

Check post-creazione:
- Nessun duplicato in word2idx
- len(word2idx) = len(idx2word) = 50001
- idx2word[0] = "UNK"
- Sum(word_freq.values()) = totale occorrenze corpus
- Top parole sono funzionali (the, of, and, ...)

## Estensione

Per aumentare vocabolario:
1. Modificare VOCAB_SIZE in config.py
2. Re-run build_vocab.py
3. Re-train tutti modelli (embeddings dimension cambia)

Trade-off:
- Vocab più grande: maggiore copertura, più parametri, training più lento
- Vocab più piccolo: meno parametri, training veloce, più UNK
