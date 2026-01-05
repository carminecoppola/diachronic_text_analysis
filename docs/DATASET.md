# Dataset: Google Books N-grams v3

## Panoramica

**Google Books N-grams v3 (2020)**: corpus derivato da ~8% di tutti i libri pubblicati, con frequenze di parole dal 1500 al 2019.

**Periodo analizzato**: 1900-1990 (91 anni, 10 decenni)

---

## Struttura Dati

### Posizione File

Storage esterno (symlink):
```
data/ → /storage/external_01/diachronic_text_analysis/data/
├── raw/           # ~27GB (file originali + filtrato)
└── processed/     # 3.1GB (10 decenni + vocabolario)
```

---

## File del Dataset

### 1. File Originali (`data/raw/*.gz`)

**24 file compressi** da Google Cloud Storage:
- `1-00000-of-00024.gz` → `1-00023-of-00024.gz`
- ~13GB totali compressi
- Copertura: 1500-2019 (tutti gli anni)

**Formato V3 compatto**:
```
word    anno1,count1,vol1    anno2,count2,vol2    ...
```

**Esempio**:
```
computer    1850,2,1    1900,50,10    1990,100000,5000
```
- `anno,occorrenze,numero_libri`

**Divisione alfabetica**:
- File 0-5: caratteri speciali/numeri → scartati
- File 6-23: parole alfabetiche (A-Z) → usati

---

### 2. File Filtrato (`data/raw/1gram_filtered.tsv`)

**Output preprocessing**: `download_ngrams.py`
- **766 milioni di righe**
- **14GB non compresso**
- Solo anni 1900-1990
- Solo parole alfabetiche

**Formato esploso** (1 riga per anno):
```
word    year    match_count    volume_count
```

**Esempio**:
```
computer    1900    50      10
computer    1901    60      12
...
computer    1990    100000  5000
```

**Filtri**:
- Anno ∈ [1900, 1990]
- `word.isalpha() == True`
- Lunghezza ≥ 2 caratteri

---

### 3. File Processati (`data/processed/*.txt`)

**10 file decennali** (output `preprocess.py`):
- `1900s.txt`, `1910s.txt`, ..., `1990s.txt`
- **3.1GB totali** (281-378MB per file)
- **113.7M righe totali**

**Statistiche**:
| Decennio | Parole uniche | Dimensione |
|----------|---------------|------------|
| 1900s | 10,368,304 | 281MB |
| 1910s | 10,643,476 | 289MB |
| 1920s | 10,554,137 | 287MB |
| 1930s | 10,326,691 | 281MB |
| 1940s | 10,504,289 | 286MB |
| 1950s | 11,845,648 | 323MB |
| 1960s | 13,338,122 | 364MB |
| 1970s | 13,798,580 | 378MB |
| 1980s | 13,740,208 | 376MB |
| 1990s | 8,638,230  | 236MB |

**Crescita**: 10.3M (1900s) → 13.7M (1980s) parole uniche

**Formato**:
```
word    normalized_frequency
```

**Esempio** (`1950s.txt`):
```
the         5.234567e-02
computer    5.678901e-07
atomic      4.567890e-07
```

**Normalizzazione**:
```python
freq = count_1950s / total_words_1950s
```
Rende comparabili decenni con volumi diversi.

---

### 4. Vocabolario (`data/processed/vocab.json`)

**Output**: `build_vocab.py`
- **50,001 parole** (top 50K + `<UNK>`)
- **990KB** (JSON) + 1.6KB (statistiche)
- **Copertura**: 99.98-100% per decennio

Dettagli: [VOCABOLARIO.md](VOCABOLARIO.md)

---

## Pipeline Trasformazione

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  ORIGINALE  │  →   │  FILTRATO   │  →   │ PROCESSATO  │
│  24 file gz │      │  1 file TSV │      │  10 file    │
├─────────────┤      ├─────────────┤      ├─────────────┤
│ 1500-2019   │      │ 1900-1990   │      │ 10 decenni  │
│ Compatto    │      │ Esploso     │      │ Frequenze   │
│ Tutte       │      │ Alfabetiche │      │ aggregate   │
└─────────────┘      └─────────────┘      └─────────────┘
    ~13GB                 14GB                 3.1GB
```

---

## Statistiche Complessive

| Metrica | Valore |
|---------|--------|
| File originali scaricati | 24/24 (100%) |
| Righe filtrate | 766,654,269 |
| Dimensione filtrata | 14GB |
| Decenni analizzati | 10 (1900s-1990s) |
| Parole uniche totali | 15,075,614 |
| Parole uniche/decennio | 8.6M - 13.8M |
| File processati | 3.1GB (10 file) |
| Righe processate totali | 113,757,685 |
| Anni coperti | 91 (1900-1990) |
| Copertura corpus | ~8% libri pubblicati |

---

## Scelte di Design

### Perché 1900-1990?

**Pre-1900**: Ortografia instabile, meno testi, OCR bassa qualità
**1900-1990**: Secolo stabile, massimi cambiamenti semantici
**Post-1990**: Coverage incompleta, decade 1990s parziale

### Perché decenni (non anni)?

1. **Riduce rumore**: fluttuazioni anno-anno insignificanti
2. **Cattura trend**: drift semantico è graduale (10+ anni)
3. **Gestibile**: 10 modelli vs 91
4. **Naturale**: decade = unità socio-culturale

---

## Fonte

- **Dataset**: Google Books N-grams Version 3 (2020)
- **URL**: http://storage.googleapis.com/books/ngrams/books/20200217/eng
- **Lingua**: English (`eng`)
- **Formato**: 1-grams (singole parole)

---

## Utilizzo nel Progetto

1. **Vocabolario comune** → `vocab.json` (50K parole)
2. **Training embeddings** → 1 modello CBOW per decennio
3. **Analisi semantic drift** → confronto tra decenni
4. **Visualizzazione** → PCA/t-SNE traiettorie parole

---

## Script Generazione

| Fase | Script | Input | Output | Tempo |
|------|--------|-------|--------|-------|
| 1 | `download_ngrams.py` | Google Cloud | `1gram_filtered.tsv` (14GB) | 4-6h |
| 2 | `preprocess.py` | Filtrato | 10 file decenni (3.1GB) | 30-45m |
| 3 | `build_vocab.py` | Decenni | `vocab.json` (990KB) | 5-10m |

---

## Comandi Utili

```bash
# Esplora file processato
head -20 data/processed/1950s.txt

# Conta parole uniche per decennio
wc -l data/processed/*.txt

# Visualizza statistiche vocabolario
cat data/processed/vocab_stats.txt

# Verifica dimensioni
du -sh data/raw/ data/processed/
```

---

## FAQ

**Q: Perché storage esterno (symlink)?**
A: Dati pesanti (17GB) su disco dedicato, codice su workspace principale.

**Q: Posso scaricare solo subset?**
A: Sì, modifica `NUM_FILES_TO_DOWNLOAD` in `config.py` (default 24, test con 3).

**Q: Formato "esploso" vs "compatto"?**
A: Compatto: 1 riga con tutti gli anni. Esploso: 1 riga per anno. Più facile da processare.

**Q: Normalizzazione frequenze necessaria?**
A: Sì, altrimenti decenni con più libri dominerebbero (bias volumetrico).

**Q: Perché solo parole alfabetiche?**
A: Numeri e simboli spesso sono errori OCR o non linguisticamente rilevanti.

---

**Ultimo aggiornamento**: 5 Gennaio 2026
**Dettagli vocabolario**: [VOCABOLARIO.md](VOCABOLARIO.md)
**Spiegazione progetto**: [SPIEGAZIONE_PROGETTO.md](SPIEGAZIONE_PROGETTO.md)
