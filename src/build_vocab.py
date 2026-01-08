"""
FASE 3: Costruzione vocabolario comune

Questo script costruisce un vocabolario COMUNE a tutti i decenni.

COSA FA:
1. Legge tutti i file processed (1900s.txt, ..., 1990s.txt)
2. Estrae parole singole dai 3-gram ("word1 word2 word3" → 3 parole)
3. Calcola frequenza totale di ogni parola (somma su tutti i decenni)
4. Seleziona top-K parole più frequenti
5. Crea mapping word → index
6. Salva vocabolario in formato JSON

PERCHÉ VOCABOLARIO COMUNE:
Per confrontare embeddings tra decenni, serve che le stesse parole
siano presenti in tutti i periodi. Parole che appaiono solo in un
decennio non possono essere confrontate.

INPUT:
- data/processed/*.txt (tutti i decenni, formato: "word1 word2 word3\tfreq")

OUTPUT:
- data/processed/vocab.json: {word: index}
- data/processed/vocab_stats.txt: statistiche vocabolario

Formato vocab.json:
{
  "<UNK>": 0,
  "the": 1,
  "of": 2,
  ...
}"""
import os
import sys
import json
from collections import defaultdict
from typing import Dict, List, Tuple

# Import config
from config import (
    DATA_PROCESSED_DIR,
    VOCAB_FILE,
    VOCAB_SIZE,
    MIN_VOCAB_FREQ,
    UNK_TOKEN,
    DECADES,
    get_processed_file_path,
    create_directories
)


def load_decade_frequencies(decade: str) -> Dict[str, float]:
    """
    Carica frequenze di un decennio estraendo parole singole dai 3-gram.
    
    Args:
        decade: Nome decennio (es. "1990s")
    
    Returns:
        Dizionario {word: frequency} con parole singole
    """
    file_path = get_processed_file_path(decade)
    
    word_freq = defaultdict(float)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                ngram, freq_str = parts
                freq = float(freq_str)
                
                # Estrai le 3 parole dal 3-gram
                words = ngram.split()
                
                # Ogni parola nel 3-gram contribuisce alla sua frequenza
                # Dividiamo equamente la frequenza tra le 3 parole
                for word in words:
                    word_freq[word] += freq / len(words)
    
    return dict(word_freq)


def compute_total_frequencies() -> Dict[str, float]:
    """
    Calcola frequenze totali (somma su tutti i decenni).
    
    Returns:
        Dizionario {word: total_frequency}
    """
    print("Caricamento frequenze da tutti i decenni...")
    
    total_freq = defaultdict(float)
    
    for decade in DECADES:
        print(f"  Caricamento {decade}...")
        decade_freq = load_decade_frequencies(decade)
        
        for word, freq in decade_freq.items():
            total_freq[word] += freq
        
        print(f"    {len(decade_freq):,} parole")
    
    print(f"\n✓ Parole uniche totali: {len(total_freq):,}")
    
    return dict(total_freq)


def build_vocabulary(total_freq: Dict[str, float]) -> Dict[str, int]:
    """
    Costruisce vocabolario selezionando top-K parole.
    
    Args:
        total_freq: Frequenze totali
    
    Returns:
        Vocabolario {word: index}
    """
    print(f"\nCostruzione vocabolario (top {VOCAB_SIZE} parole)...")
    
    # Ordina parole per frequenza decrescente
    sorted_words = sorted(
        total_freq.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Seleziona top-K parole più frequenti
    # Da ~15M parole uniche → prendiamo solo le 50K più comuni e stabili
    selected_words = sorted_words[:VOCAB_SIZE]
    
    # Crea vocabolario: mapping parola → indice numerico
    # Indice 0 è riservato per <UNK> (parole sconosciute/fuori vocabolario)
    # Questo sarà usato per convertire parole in indici durante il training
    vocab = {UNK_TOKEN: 0}  # Token speciale per parole sconosciute
    
    # Assegna indici 1, 2, 3, ... alle parole in ordine di frequenza
    # Esempio: {"<UNK>": 0, "the": 1, "of": 2, "and": 3, ...}
    for idx, (word, freq) in enumerate(selected_words, start=1):
        vocab[word] = idx
    
    print(f"✓ Vocabolario costruito: {len(vocab):,} parole (incluso {UNK_TOKEN})")
    
    return vocab


def save_vocabulary(vocab: Dict[str, int], total_freq: Dict[str, float]):
    """
    Salva vocabolario e statistiche.
    
    Args:
        vocab: Vocabolario {word: index}
        total_freq: Frequenze totali
    """
    print(f"\nSalvataggio vocabolario in {VOCAB_FILE}...")
    
    # Salva vocab JSON
    with open(VOCAB_FILE, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Vocabolario salvato: {len(vocab):,} parole")
    
    # Salva statistiche
    stats_file = VOCAB_FILE.replace('.json', '_stats.txt')
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("VOCABOLARIO - STATISTICHE\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Dimensione vocabolario: {len(vocab):,}\n")
        f.write(f"Token speciale UNK: {UNK_TOKEN}\n\n")
        
        # Top 20 parole
        f.write("Top 20 parole più frequenti:\n")
        f.write("-" * 50 + "\n")
        
        sorted_vocab = sorted(
            [(w, i) for w, i in vocab.items() if w != UNK_TOKEN],
            key=lambda x: x[1]
        )
        
        for word, idx in sorted_vocab[:20]:
            freq = total_freq.get(word, 0)
            f.write(f"{idx:6d}. {word:20s} {freq:.6e}\n")
        
        f.write("\n")
        
        # Statistiche copertura per decennio
        f.write("Copertura vocabolario per decennio:\n")
        f.write("-" * 50 + "\n")
        
        for decade in DECADES:
            decade_freq = load_decade_frequencies(decade)
            
            # Quante parole del vocabolario sono presenti?
            present = sum(1 for w in vocab.keys() if w != UNK_TOKEN and w in decade_freq)
            coverage = 100 * present / (len(vocab) - 1)  # -1 per UNK
            
            f.write(f"{decade}: {present:,}/{len(vocab)-1:,} parole ({coverage:.2f}%)\n")
    
    print(f"✓ Statistiche salvate in {stats_file}")


def main():
    # Main function.
    
    print("="*70)
    print("COSTRUZIONE VOCABOLARIO COMUNE")
    print("="*70)
    print(f"Dimensione target: {VOCAB_SIZE:,} parole")
    print(f"Decenni: {', '.join(DECADES)}")
    print("="*70 + "\n")
    
    # Verifica file input
    for decade in DECADES:
        file_path = get_processed_file_path(decade)
        if not os.path.exists(file_path):
            print(f"❌ File mancante: {file_path}")
            print("Eseguire prima preprocess.py (FASE 2)")
            return 1
    
    # Step 1: Calcola frequenze totali
    total_freq = compute_total_frequencies()
    
    # Step 2: Costruisci vocabolario
    vocab = build_vocabulary(total_freq)
    
    # Step 3: Salva
    save_vocabulary(vocab, total_freq)
    
    print("\n" + "="*70)
    print("✅ FASE 3 COMPLETATA")
    print("="*70)
    print(f"Vocabolario salvato in: {VOCAB_FILE}")
    print("\nProssimi step:")
    print("  - Implementare dataset.py (PyTorch Dataset)")
    print("  - Implementare model.py (CBOW)")
    print("  - Implementare train.py (training embeddings)")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())  