# Vocabolario Comune

## Cos'è?

Un **dizionario che mappa ogni parola a un numero univoco (indice)**, necessario per convertire testo in formato numerico per machine learning.

**Esempio**:
```json
{
  "<UNK>": 0,
  "the": 1,
  "of": 2,
  "and": 3,
  "computer": 1523,
  "gay": 2891
}
```

---

## A Cosa Serve?

### 1. Conversione Testo → Numeri

**Prima**: `"the computer is running"`
**Dopo**: `[1, 1523, 45, 892]`

### 2. Vocabolario Comune Tra Decenni

Per confrontare "computer" negli anni '50 vs '90, **serve stesso indice** in entrambi i periodi:
- Con vocab comune: "computer" = 1523 in tutti i decenni → comparabile
- Senza: "computer" = 234 (anni '50), 1823 (anni '90) → non comparabile

### 3. Gestione Parole Sconosciute

Token `<UNK>` (indice 0) gestisce:
- Parole troppo rare (fuori top 50K)
- Errori di spelling
- Parole nuove durante inferenza

---

## Come Viene Creato?

**Script**: [`src/build_vocab.py`](../src/build_vocab.py)

```
1. Carica frequenze da 10 decenni
   → 1900s: 10.3M parole | 1910s: 10.6M | ... | 1990s: 8.6M

2. Aggrega frequenze totali
   → 15.075.614 parole uniche attraverso il secolo

3. Ordina per frequenza decrescente
   → "the": 56B occorrenze | "of": 35B | "and": 23B | ...

4. Seleziona top 50.000 parole
   → Cattura parole comuni e stabili nel tempo

5. Crea mapping parola → indice
   → {<UNK>: 0, the: 1, of: 2, and: 3, ...}

6. Salva JSON + statistiche
```

**Tempo esecuzione**: ~5-10 minuti

---

## File Generati

### 1. vocab.json (990KB)

**Percorso**: `data/processed/vocab.json`

**Utilizzo**:
```python
import json

with open('data/processed/vocab.json', 'r') as f:
    vocab = json.load(f)

# Converti parola → indice
word = "computer"
idx = vocab.get(word, 0)  # 0 se non trovata (<UNK>)
print(f"{word} → {idx}")  # computer → 1523

# Converti frase → sequenza
sentence = "the computer is running"
indices = [vocab.get(w, 0) for w in sentence.split()]
print(indices)  # [1, 1523, 6, 892]
```

### 2. vocab_stats.txt (1.6KB)

**Percorso**: `data/processed/vocab_stats.txt`

**Contenuto**:
```
Dimensione vocabolario: 50,001
Token speciale UNK: <UNK>

Top 20 parole più frequenti:
     1. the        5.623378e+10
     2. of         3.554629e+10
     3. and        2.338277e+10
     ...

Copertura vocabolario per decennio:
1900s: 49,999/50,000 (100.00%)
1910s: 49,989/50,000 (99.98%)
1940s: 50,000/50,000 (100.00%)
...
```

**Interpretazione**: 99.98-100% copertura = parole estremamente stabili nel tempo!

---

## Statistiche Chiave

| Metrica | Valore | Nota |
|---------|--------|------|
| **Parole uniche totali** | 15.075.614 | Tutti i decenni |
| **Vocabolario finale** | 50.001 | Top 50K + `<UNK>` |
| **Copertura per decennio** | 99.98-100% | Stabilità eccellente |
| **Dimensione file** | 990KB | Veloce da caricare |
| **Parola più frequente** | "the" (56B) | Articolo inglese |

---

## Parametri (`config.py`)

```python
VOCAB_SIZE = 50000       # Top-K parole più frequenti
MIN_VOCAB_FREQ = 100     # Soglia minima (filtra errori OCR)
UNK_TOKEN = "<UNK>"      # Token per parole sconosciute
```

**Perché 50.000?**
- Copre parole comuni (99.98%+ copertura)
- Gestibile computazionalmente (50K×300 = 15M parametri)
- Oltre 50K aggiungi soprattutto nomi propri/errori

---

## Esempi Pratici

### Verificare Presenza Parola

```python
test_words = ["computer", "internet", "smartphone", "gay"]

for word in test_words:
    if word in vocab:
        print(f"'{word}' → indice {vocab[word]}")
    else:
        print(f"'{word}' → <UNK>")
```

**Output**:
```
'computer' → indice 1523
'internet' → indice 15234
'smartphone' → <UNK>
'gay' → indice 2891
```

### Reverse Mapping (Indice → Parola)

```python
# Crea mapping inverso
idx_to_word = {idx: word for word, idx in vocab.items()}

# Decodifica
indices = [1, 1523, 6, 892]
decoded = " ".join([idx_to_word[idx] for idx in indices])
print(decoded)  # "the computer is running"
```

---

## Come Rigenerare

```bash
source .venv/bin/activate
python src/build_vocab.py
```

**Output atteso**:
```
COSTRUZIONE VOCABOLARIO COMUNE
Caricamento frequenze da tutti i decenni...
  1900s... 10,368,304 parole
  1910s... 10,643,476 parole
  ...
Parole uniche totali: 15,075,614
Vocabolario costruito: 50,001 parole
Salvato in: data/processed/vocab.json
FASE 3 COMPLETATA
```

---

## File Correlati

- **Generato da**: [`src/build_vocab.py`](../src/build_vocab.py)
- **Configurazione**: [`src/config.py`](../src/config.py)
- **Input**: `data/processed/*.txt` (output `preprocess.py`)
- **Usato da**: `train.py` (prossima fase)

---

## FAQ

**Q: Perché comune a tutti i decenni?**
A: Per confrontare significato parole nel tempo, serve stesso indice sempre.

**Q: Cosa succede a parole fuori vocabolario?**
A: Mappate a indice 0 (`<UNK>`). Durante training, `<UNK>` impara embedding generico.

**Q: Posso usare vocab più grande (100K)?**
A: Sì, modifica `VOCAB_SIZE` in `config.py`. Ma oltre 50K aggiungi poco valore (nomi propri/errori).

**Q: Perché "the" così frequente?**
A: È l'articolo inglese più usato. Normale (Legge di Zipf): poche parole dominano.

**Q: File JSON è efficiente?**
A: Sì, 990KB si carica in millisecondi. JSON è leggibile e portabile.

---

**Creato**: 5 Gennaio 2026
**Fase**: 3/10 (Completata)
**Dettagli dataset**: [DATASET.md](DATASET.md)
