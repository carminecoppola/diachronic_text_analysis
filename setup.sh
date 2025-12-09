#!/bin/bash

"""
Script di Setup Automatico per Progetto Diachronic Text Analysis

Questo script automatizza:
1. Creazione ambiente virtuale
2. Installazione dipendenze
3. Download risorse NLP
"""

echo "==================================================================="
echo "  Setup Progetto: Diachronic Text Analysis"
echo "==================================================================="
echo ""

# Verifica versione Python
echo "[1/5] Verifica versione Python..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ ERRORE: Python 3 non trovato. Installare Python 3.8+"
    exit 1
fi
echo "✓ Python trovato"
echo ""

# Crea ambiente virtuale
echo "[2/5] Creazione ambiente virtuale .venv..."
if [ -d ".venv" ]; then
    echo "⚠️  Ambiente virtuale già esistente. Vuoi rimuoverlo? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        rm -rf .venv
        echo "✓ Ambiente virtuale rimosso"
    else
        echo "✓ Utilizzo ambiente esistente"
    fi
fi

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✓ Ambiente virtuale creato"
fi
echo ""

# Attiva ambiente virtuale
echo "[3/5] Attivazione ambiente virtuale..."
source .venv/bin/activate
echo "✓ Ambiente virtuale attivo"
echo ""

# Aggiorna pip
echo "[4/5] Aggiornamento pip..."
pip install --upgrade pip --quiet
echo "✓ pip aggiornato"
echo ""

# Installa dipendenze
echo "[5/5] Installazione dipendenze da requirements.txt..."
pip install -r requirements.txt
echo "✓ Dipendenze installate"
echo ""

# Download risorse NLP
echo "[BONUS] Download risorse NLP..."
echo "  - NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); print('✓ NLTK resources downloaded')"

echo "  - spaCy model..."
python -m spacy download en_core_web_sm --quiet
echo "✓ spaCy model downloaded"
echo ""

echo "==================================================================="
echo "  ✅ SETUP COMPLETATO!"
echo "==================================================================="
echo ""
echo "Per attivare l'ambiente virtuale in futuro:"
echo "  Linux/macOS: source .venv/bin/activate"
echo "  Windows:     .\.venv\Scripts\activate"
echo ""
echo "Prossimi passi:"
echo "  1. Scaricare dataset storico in data/raw/"
echo "  2. Eseguire preprocessing: python src/preprocess.py"
echo "  3. Costruire vocabolario: python src/build_vocab.py"
echo "  4. Vedere README.md per pipeline completa"
echo ""
