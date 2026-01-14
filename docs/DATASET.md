# Dataset e Preprocessing

## Google Books N-grams

Dataset: Google Books N-grams versione 3 (2020)
- Lingua: inglese
- Periodo: 1900-2019
- Tipo: 3-gram e 5-gram
- Fonte: http://storage.googleapis.com/books/ngrams/books

### Struttura N-gram

```
word1 word2 word3    year    match_count    volume_count
the computer is      1990    1543          87
```

### Filtering

Criteri durante download:
- Anno: 1900-2019
- Match count >= 100
- Solo lowercase alfabetiche
- Rimossi underscore (POS tags)

## Preprocessing

### Aggregazione per Decennio

Script: `src/data/preprocess.py`

Processo:
1. Raggruppa anni in decenni (1900-1909 → 1900s)
2. Somma frequenze per n-gram identico
3. Mantiene n-gram completi (non singole parole)

Output: `data/processed/{3gram,5gram}/{decade}.txt`

Formato:
```
word1 word2 word3\tfreq
the computer is\t15430
```

### Vocabolario Comune

Script: `src/data/build_vocab.py`

Processo:
1. Estrae parole singole da tutti n-gram
2. Conta frequenza totale cross-decade
3. Seleziona top 50K + token UNK

Output: `data/processed/{3gram,5gram}/vocab.json`

Struttura:
```json
{
  "word2idx": {"<UNK>": 0, "the": 1, ...},
  "idx2word": ["<UNK>", "the", ...],
  "word_freq": {"the": 123456789, ...}
}
```

### Statistiche

3-gram:
- Raw: ~15GB
- Processed: ~12GB (12 decenni)
- Vocabolario: 50,001 parole
- N-gram unici per decade: ~100M

5-gram:
- Raw: ~20GB
- Processed: ~15GB (12 decenni)
- Vocabolario: 50,001 parole (stesso)
- N-gram unici per decade: ~150M

## Considerations

### N-gram completi

Mantenerli preserva contesto originale:
- "computer is running" vs "program is running" = relazioni diverse
- Contesto reale da corpus (non finestre artificiali)
- Struttura linguistica e sintassi preservate

### Threshold Frequenza

Min count 100:
- Bilancia copertura e qualità
- Rimuove typos e OCR errors
- Mantiene parole rare ma reali
- Riduce dimensione (~30%)

### Vocabolario Condiviso

Stesso per tutti decenni:
- Confronto diretto embeddings
- Parole nuove → UNK
- Focus su parole stabili
