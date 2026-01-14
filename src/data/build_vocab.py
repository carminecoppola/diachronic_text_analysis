"""
FASE 3: Costruzione vocabolario comune (interattivo)

Questo script costruisce un vocabolario COMUNE a tutti i decenni per una directory di dati processati scelta dall'utente.
"""
import os
import sys
import json
import glob
from collections import defaultdict
from typing import Dict

from src.config import VOCAB_SIZE, UNK_TOKEN

def get_decade_files(processed_dir):
    """Restituisce lista di file decennali (1900s.txt, ...) in una directory."""
    files = sorted(glob.glob(os.path.join(processed_dir, "[0-9][0-9][0-9]0s.txt")))
    return files

def load_decade_frequencies(file_path: str) -> Dict[str, int]:
    """Carica frequenze parole da un file decennale."""
    word_freq = defaultdict(int)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            ngram_text = line.strip()
            if ngram_text:
                for word in ngram_text.split():
                    word_freq[word] += 1
    return dict(word_freq)

def compute_total_frequencies(decade_files) -> Dict[str, int]:
    """Calcola frequenze totali sommando tutti i file decennali."""
    print("Caricamento frequenze da tutti i decenni...")
    total_freq = defaultdict(int)
    for file_path in decade_files:
        decade = os.path.basename(file_path).replace('.txt','')
        print(f"  {decade}...", end=" ")
        decade_freq = load_decade_frequencies(file_path)
        for word, freq in decade_freq.items():
            total_freq[word] += freq
        print(f"{len(decade_freq):,} parole")
    print(f"\n✓ Parole uniche totali: {len(total_freq):,}")
    return dict(total_freq)

def build_vocabulary(total_freq: Dict[str, int]) -> Dict[str, int]:
    """Costruisce vocabolario: top-K parole → indici."""
    print(f"\nCostruzione vocabolario (top {VOCAB_SIZE})...")
    sorted_words = sorted(total_freq.items(), key=lambda x: x[1], reverse=True)
    selected = sorted_words[:VOCAB_SIZE]
    vocab = {UNK_TOKEN: 0}
    for idx, (word, _) in enumerate(selected, start=1):
        vocab[word] = idx
    print(f"✓ Vocabolario: {len(vocab):,} parole (incluso {UNK_TOKEN})")
    return vocab

def save_vocabulary(vocab: Dict[str, int], total_freq: Dict[str, int], out_dir: str, decade_files):
    """Salva vocabolario JSON e statistiche nella directory scelta."""
    vocab_file = os.path.join(out_dir, "vocab.json")
    print(f"\nSalvataggio {vocab_file}...")
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"✓ Salvato: {len(vocab):,} parole")
    stats_file = vocab_file.replace('.json', '_stats.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("VOCABOLARIO - STATISTICHE\n")
        f.write("="*70 + "\n\n")
        f.write(f"Dimensione: {len(vocab):,}\n")
        f.write(f"UNK token: {UNK_TOKEN}\n\n")
        # Top 20
        f.write("Top 20 parole:\n" + "-"*50 + "\n")
        sorted_vocab = sorted([(w, i) for w, i in vocab.items() if w != UNK_TOKEN], key=lambda x: x[1])
        for word, idx in sorted_vocab[:20]:
            f.write(f"{idx:6d}. {word:20s} {total_freq.get(word, 0):.6e}\n")
        # Copertura per decennio
        f.write("\nCopertura per decennio:\n" + "-"*50 + "\n")
        for file_path in decade_files:
            decade = os.path.basename(file_path).replace('.txt','')
            decade_freq = load_decade_frequencies(file_path)
            present = sum(1 for w in vocab if w != UNK_TOKEN and w in decade_freq)
            coverage = 100 * present / (len(vocab) - 1)
            f.write(f"{decade}: {present:,}/{len(vocab)-1:,} ({coverage:.2f}%)\n")
    print(f"✓ Statistiche: {stats_file}")

def main():
    print("="*70)
    print("COSTRUZIONE VOCABOLARIO COMUNE")
    print("="*70)
    print(f"Dimensione target: {VOCAB_SIZE:,}")
    # Trova tutte le directory processed con file decennali
    base_dir = os.path.join("data", "processed")
    candidates = []
    for d in os.listdir(base_dir):
        full = os.path.join(base_dir, d)
        if os.path.isdir(full):
            files = get_decade_files(full)
            if len(files) >= 3:
                candidates.append((d, full, files))
    if not candidates:
        print("❌ Nessuna directory con dati decennali trovata in data/processed/")
        return 1
    print("\nOpzioni disponibili:")
    for i, (d, full, files) in enumerate(candidates):
        print(f"  [{i+1}] {d} ({len(files)} decenni)")
    scelta = input(f"\nScegli la directory [1-{len(candidates)}]: ").strip()
    try:
        idx = int(scelta) - 1
        assert 0 <= idx < len(candidates)
    except Exception:
        print("Scelta non valida.")
        return 1
    d, out_dir, decade_files = candidates[idx]
    print(f"\nUserò: {out_dir}")
    total_freq = compute_total_frequencies(decade_files)
    vocab = build_vocabulary(total_freq)
    save_vocabulary(vocab, total_freq, out_dir, decade_files)
    print("\n" + "="*70)
    print("COMPLETATO")
    print("="*70)
    print(f"Vocabolario: {os.path.join(out_dir, 'vocab.json')}")
    print("\nProssimo: train.py (training embeddings)")
    print("="*70)
    return 0

if __name__ == "__main__":
    sys.exit(main())