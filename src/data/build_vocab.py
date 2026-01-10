"""
FASE 3: Costruzione vocabolario comune

Questo script costruisce un vocabolario COMUNE a tutti i decenni.

COSA FA:
1. Legge tutti i file processed (1900s.txt, ..., 2010s.txt)
2. Estrae parole singole dagli n-gram ("word1 word2..." → N parole)
3. Calcola frequenza totale di ogni parola (somma su tutti i decenni)
4. Seleziona top-K parole più frequenti
5. Crea mapping word → index
6. Salva vocabolario in formato JSON

PERCHÉ VOCABOLARIO COMUNE:
Per confrontare embeddings tra decenni, serve che le stesse parole
siano presenti in tutti i periodi. Parole che appaiono solo in un
decennio non possono essere confrontate.

INPUT:
- data/processed/{NGRAM_TYPE}gram/*.txt (tutti i decenni, formato: "word1 word2...\tfreq")

OUTPUT:
- data/processed/{NGRAM_TYPE}gram/vocab.json: {word: index}
- data/processed/{NGRAM_TYPE}gram/vocab_stats.txt: statistiche vocabolario

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
from src.config import (VOCAB_FILE, VOCAB_SIZE, UNK_TOKEN, DECADES, 
                    get_processed_file_path)


def load_decade_frequencies(decade: str) -> Dict[str, int]:
    """
    Carica frequenze parole da un decennio.
    Formato: ogni riga = un n-gram, ogni parola conta come 1 occorrenza.
    """
    word_freq = defaultdict(int)
    
    with open(get_processed_file_path(decade), 'r', encoding='utf-8') as f:
        for line in f:
            ngram_text = line.strip()
            if ngram_text:
                for word in ngram_text.split():
                    word_freq[word] += 1
    
    return dict(word_freq)


def compute_total_frequencies() -> Dict[str, int]:
    """Calcola frequenze totali sommando tutti i decenni."""
    print("Caricamento frequenze da tutti i decenni...")
    total_freq = defaultdict(int)
    
    for decade in DECADES:
        print(f"  {decade}...", end=" ")
        decade_freq = load_decade_frequencies(decade)
        for word, freq in decade_freq.items():
            total_freq[word] += freq
        print(f"{len(decade_freq):,} parole")
    
    print(f"\n✓ Parole uniche totali: {len(total_freq):,}")
    return dict(total_freq)


def build_vocabulary(total_freq: Dict[str, int]) -> Dict[str, int]:
    """Costruisce vocabolario: top-K parole → indici."""
    print(f"\nCostruzione vocabolario (top {VOCAB_SIZE})...")
    
    # Ordina per frequenza, seleziona top-K
    sorted_words = sorted(total_freq.items(), key=lambda x: x[1], reverse=True)
    selected = sorted_words[:VOCAB_SIZE]
    
    # Crea mapping {word: index}, indice 0 riservato per <UNK>
    vocab = {UNK_TOKEN: 0}
    for idx, (word, _) in enumerate(selected, start=1):
        vocab[word] = idx
    
    print(f"✓ Vocabolario: {len(vocab):,} parole (incluso {UNK_TOKEN})")
    return vocab


def save_vocabulary(vocab: Dict[str, int], total_freq: Dict[str, int]):
    """Salva vocabolario JSON e statistiche."""
    print(f"\nSalvataggio {VOCAB_FILE}...")
    
    # Salva JSON
    with open(VOCAB_FILE, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"✓ Salvato: {len(vocab):,} parole")
    
    # Salva statistiche
    stats_file = VOCAB_FILE.replace('.json', '_stats.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("VOCABOLARIO - STATISTICHE\n")
        f.write("="*70 + "\n\n")
        f.write(f"Dimensione: {len(vocab):,}\n")
        f.write(f"UNK token: {UNK_TOKEN}\n\n")
        
        # Top 20
        f.write("Top 20 parole:\n" + "-"*50 + "\n")
        sorted_vocab = sorted([(w, i) for w, i in vocab.items() if w != UNK_TOKEN], 
                             key=lambda x: x[1])
        for word, idx in sorted_vocab[:20]:
            f.write(f"{idx:6d}. {word:20s} {total_freq.get(word, 0):.6e}\n")
        
        # Copertura per decennio
        f.write("\nCopertura per decennio:\n" + "-"*50 + "\n")
        for decade in DECADES:
            decade_freq = load_decade_frequencies(decade)
            present = sum(1 for w in vocab if w != UNK_TOKEN and w in decade_freq)
            coverage = 100 * present / (len(vocab) - 1)
            f.write(f"{decade}: {present:,}/{len(vocab)-1:,} ({coverage:.2f}%)\n")
    
    print(f"✓ Statistiche: {stats_file}")


def main():
    """Main: costruisce vocabolario comune."""
    print("="*70)
    print("COSTRUZIONE VOCABOLARIO COMUNE")
    print("="*70)
    print(f"Dimensione target: {VOCAB_SIZE:,}")
    print(f"Decenni: {', '.join(DECADES)}")
    print("="*70 + "\n")
    
    # Verifica file input
    for decade in DECADES:
        if not os.path.exists(get_processed_file_path(decade)):
            print(f"❌ File mancante: {decade}.txt")
            print("Eseguire prima preprocess.py")
            return 1
    
    # Pipeline
    total_freq = compute_total_frequencies()
    vocab = build_vocabulary(total_freq)
    save_vocabulary(vocab, total_freq)
    
    print("\n" + "="*70)
    print("✅ COMPLETATO")
    print("="*70)
    print(f"Vocabolario: {VOCAB_FILE}")
    print("\nProssimo: train.py (training embeddings)")
    print("="*70)
    return 0


if __name__ == "__main__":
    sys.exit(main())  