# 📋 SPIEGAZIONE PROGETTO - Diachronic Text Analysis

## Cos'è `.gitkeep`? (E perché l'abbiamo rimosso)

`.gitkeep` è un file **placeholder** vuoto usato per "tracciare" cartelle vuote in Git.

**Problema**: Git NON versiona cartelle vuote.

**Soluzione vecchia**:
```
data/raw/.gitkeep    # Forza Git a tracciare data/raw/ anche se vuota
models/.gitkeep      # Idem per models/
```

**Soluzione migliore** (quella che usiamo ora):
- Mettiamo nel `.gitignore` **l'intera cartella** `data/`, `models/`, ecc.
- Tanto questi file sono:
  - Troppo grandi per Git
  - Privati (non li vogliamo pubblicare)
  - Rigenerabili (possiamo ricrearli con gli script)

Quindi `.gitkeep` **non serve più** perché:
1. Non vogliamo versionare quelle cartelle
2. Chi clona il repo può ricrearle automaticamente con gli script

---

## Come funziona il `.gitignore` attuale?

```gitignore
# NON versionare dati (possono essere grandi o privati)
data/raw/
data/processed/

# NON versionare modelli (file pesanti, rigenerabili)
models/

# NON versionare output (rigenerabili)
alignment/
plots/

# NON versionare ambiente virtuale (specifico per ogni macchina)
.venv/
```

**Regola**: Su Git mettiamo **solo il codice** (`.py`, `.ipynb`, `README.md`).

I **dati e i risultati** li gestisce ciascuno localmente o su HPC.

---

## Cosa fa effettivamente questo progetto?

### 🎯 Obiettivo Finale

**Scoprire come le parole cambiano significato nel tempo**.

Esempio:
- "gay" nel 1900 = felice, gioioso
- "gay" nel 1990 = omosessuale

Come lo facciamo?

### 📝 Pipeline Completa (10 Fasi)

#### **FASE 0**: Setup
- Creare cartelle
- Installare dipendenze (PyTorch, NLTK, ecc.)
- ✅ **COMPLETATA**

#### **FASE 1**: Raccolta Corpus
- Trovare testi storici con date (es. libri, articoli)
- Organizzare per periodo (1900s, 1910s, ..., 1990s)
- ✅ **COMPLETATA** (con dati demo)

#### **FASE 2**: Preprocessing ← **QUI SIAMO**
- **Input**: Testi grezzi in `data/raw/`
- **Pipeline**:
  1. Lowercase: "Hello" → "hello"
  2. Rimuovi punteggiatura: "hello, world!" → "hello world"
  3. Tokenizzazione: "hello world" → ["hello", "world"]
  4. Lemmatizzazione: "running" → "run", "mice" → "mouse"
  5. Filtra: rimuovi parole troppo corte/lunghe
- **Output**: Token puliti in `data/processed/`
- ✅ **COMPLETATA** (5M+ tokens processati)

#### **FASE 3**: Vocabolario
- **Input**: Token di tutti i periodi
- **Operazione**: 
  - Unire tutte le parole uniche
  - Assegnare un numero a ogni parola:
    ```
    {"gay": 0, "happy": 1, "cell": 2, "mouse": 3, ...}
    ```
  - Filtrare parole troppo rare (freq < 5)
- **Output**: Dizionario salvato in `data/processed/vocab.pkl`
- 🔄 **PROSSIMA**

#### **FASE 4**: Dataset PyTorch
- Creare Dataset che:
  - Legge i token
  - Li converte in numeri (usando il vocabolario)
  - Prepara "coppie" per il training CBOW
    - CBOW = predire parola centrale da contesto
    - Es: ["I", "love", "pizza", "very", "much"] → predici "pizza" da ["I", "love", "very", "much"]

#### **FASE 5**: Training Word Embeddings
- **Per ogni periodo** (1900s, 1910s, ..., 1990s):
  - Addestrare un modello CBOW
  - Input: indici parole
  - Output: vettori di 300 numeri per parola
    ```
    "gay" → [0.23, -0.45, 0.12, ..., 0.89]  (300 numeri)
    ```
- Salvare modelli in `models/emb_1900s.pt`, `models/emb_1910s.pt`, ecc.

#### **FASE 6**: Allineamento (Orthogonal Procrustes)
- **Problema**: 
  - Embeddings di periodi diversi sono in "spazi" diversi
  - Non possiamo confrontarli direttamente
- **Soluzione**:
  - Usare Orthogonal Procrustes per "allineare" gli spazi
  - È come sovrapporre due mappe con rotazione/traslazione
- **Output**: Matrici di trasformazione in `alignment/`

#### **FASE 7**: Calcolo Semantic Drift
- Per ogni parola (es. "gay"):
  - Prendi vettore_1900s: `[0.5, 0.3, ...]`
  - Prendi vettore_1990s_aligned: `[0.1, -0.8, ...]`
  - Calcola distanza coseno:
    ```
    drift = 1 - cosine_similarity(v1, v2)
    ```
  - drift = 0 → nessun cambiamento
  - drift = 1 → completamente diverso
- Output: CSV con ranking parole per drift

#### **FASE 8**: Visualizzazioni
- Usare t-SNE/PCA per ridurre 300 dimensioni → 2D
- Plottare come si spostano le parole nel tempo
- Es: "gay" vicino a "happy" nel 1900, vicino a "homosexual" nel 1990

#### **FASE 9**: Analisi e Interpretazione
- Leggere i risultati
- Identificare pattern interessanti
- Documentare casi studio

#### **FASE 10**: Ottimizzazioni
- Hyperparameter tuning
- Test con corpus diversi
- Estensioni (es. multi-lingua)

---

## 📊 Dove siamo ora?

```
[✅] Fase 0: Setup
[✅] Fase 1: Corpus Demo
[✅] Fase 2: Preprocessing  ← APPENA COMPLETATA
[🔄] Fase 3: Vocabolario    ← PROSSIMA
[ ] Fase 4-10: ...
```

### File attuali nel progetto:

```
diachronic_text_analysis/
├── .gitignore                          # Configura cosa NON versionare
├── README.md                           # Documentazione principale
├── requirements.txt                    # Lista dipendenze Python
├── .venv/                              # Ambiente virtuale (NON versionato)
│
├── data/
│   ├── raw/                            # Testi originali (NON versionato)
│   │   ├── 1900s.txt
│   │   ├── 1910s.txt
│   │   └── ... (altri periodi)
│   └── processed/                      # Token puliti (NON versionato)
│       ├── 1900s.txt                   # 502k tokens, 1 per riga
│       ├── 1910s.txt
│       └── ...
│
├── src/                                # Codice Python
│   ├── config.py                       # Tutte le configurazioni
│   ├── preprocess.py                   # Pipeline preprocessing ✅
│   ├── validate_preprocessing.py      # Validazione risultati ✅
│   ├── generate_demo_data.py          # Script per dati demo ✅
│   └── ... (altri moduli da creare)
│
├── notebooks/                          # Jupyter Notebooks
│   └── 01_intro_e_preprocessing.ipynb  # Guida interattiva ✅
│
├── models/                             # Modelli salvati (NON versionato, vuota ora)
├── alignment/                          # Matrici allineamento (NON versionato, vuota ora)
└── plots/                              # Grafici (NON versionato, vuota ora)
```

---

## 🎓 Concetti Chiave

### Word Embedding
Invece di rappresentare "gatto" come stringa, lo rappresentiamo come vettore:
```python
"gatto" → [0.23, -0.45, 0.12, 0.89, ..., 0.34]  # 300 numeri
```

**Proprietà magica**: parole simili hanno vettori vicini!
```
distance("gatto", "cane") < distance("gatto", "automobile")
```

Operazioni vettoriali hanno senso semantico:
```
king - man + woman ≈ queen
```

### CBOW (Continuous Bag of Words)
Modello per imparare embeddings:
- Input: parole di contesto → "I love [?] very much"
- Output: parola centrale → "pizza"

Il modello impara vettori tali che parole con contesti simili siano vicine.

### Orthogonal Procrustes
Problema: 
- Embeddings 1900s e 1990s sono in spazi diversi (sistemi di riferimento diversi)
- È come avere due mappe con orientamenti diversi

Soluzione:
- Trovare la rotazione/traslazione ottima per sovrapporli
- Mantiene la struttura (distanze relative) ma allinea gli assi

Matematica:
```
Minimizza: ||A @ R - B||  (trova R che sovrappone A a B)
Vincolo: R è ortogonale (solo rotazione, no stretching)
```

### Semantic Drift
Quanto è cambiata una parola?
```python
drift = 1 - cosine_similarity(vec_1900s, vec_1990s_aligned)
```

- 0 = identico significato
- 0.5 = cambiato moderatamente
- 1 = completamente diverso

---

## 💡 Esempio Pratico

### Parola: "gay"

**1900s**:
- Contesto: "happy", "cheerful", "merry", "joyful"
- Vettore vicino a parole di felicità

**1990s**:
- Contesto: "homosexual", "lesbian", "LGBT", "pride"
- Vettore vicino a parole di orientamento sessuale

**Risultato**: drift alto (es. 0.8) → significato molto cambiato!

---

## ❓ FAQ

### Q: Perché PyTorch?
A: Framework moderno per deep learning, ottimizzato GPU, flessibile.

### Q: Quanti dati servono?
A: Idealmente:
- 10M+ tokens per periodo
- 10k+ parole uniche
- 5+ periodi temporali

Il nostro demo ha ~500k tokens per semplicità didattica.

### Q: Posso usare lingue diverse dall'inglese?
A: Sì! Devi solo:
- Cambiare corpus
- Configurare lemmatizer appropriato (es. spaCy con modello italiano)

### Q: Quanto tempo ci vuole il training?
A: Dipende da:
- GPU vs CPU (GPU ~10x più veloce)
- Dimensione corpus
- Numero periodi

Stima: 10-30 minuti per periodo su GPU, 1-3 ore su CPU.

---

**Prossima azione**: Iniziare Fase 3 (costruzione vocabolario).
