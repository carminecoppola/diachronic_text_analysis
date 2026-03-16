"""
PHASE 3: Build shared vocabulary

Constructs vocabulary common across all decades from preprocessed n-gram files.
Loads word frequencies, selects top-K words, generates statistics.

Process:
1. Specify directory with decade files via --processed-dir argument
2. Aggregate word frequencies across decades
3. Build vocabulary (top-K words, including UNK token)
4. Save vocabulary.json and statistics

Vocabulary Format:
- vocab.json: {word: index} mapping where index >= 0
- UNK token has index 0, top VOCAB_SIZE-1 words have indices 1..VOCAB_SIZE-1
- Total vocabulary size: VOCAB_SIZE (including UNK)
"""
import os
import sys
import json
import glob
import argparse
from collections import defaultdict
from typing import Dict, Tuple

from src.config import VOCAB_SIZE, UNK_TOKEN, get_preprocess_output_dir

def get_decade_files(processed_dir: str):
    """Returns sorted list of decade files (1900s.txt, ...) in a directory."""
    files = sorted(glob.glob(os.path.join(processed_dir, "[0-9][0-9][0-9]0s.txt")))
    return files

def load_decade_frequencies(file_path: str) -> Dict[str, int]:
    """Loads word frequencies from a decade file."""
    word_freq = defaultdict(int)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            ngram_text = line.strip()
            if ngram_text:
                for word in ngram_text.split():
                    word_freq[word] += 1
    return dict(word_freq)

def compute_total_frequencies(decade_files) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    """
    Computes total word frequencies across all decades.
    
    Returns:
        Tuple of:
        - total_freq: aggregated frequencies across all decades
        - decade_frequencies: dict mapping decade→{word→freq} for reuse in statistics
    """
    print("  Loading frequencies from all decades...")
    total_freq = defaultdict(int)
    decade_frequencies = {}
    
    for file_path in decade_files:
        decade = os.path.basename(file_path).replace('.txt', '')
        print(f"    {decade}...", end=" ", flush=True)
        decade_freq = load_decade_frequencies(file_path)
        decade_frequencies[decade] = decade_freq  # Save for later use
        
        for word, freq in decade_freq.items():
            total_freq[word] += freq
        print(f"{len(decade_freq):,} words")
    
    print(f"\n  ✓ Total unique words: {len(total_freq):,}")
    return dict(total_freq), decade_frequencies

def build_vocabulary(total_freq: Dict[str, int]) -> Dict[str, int]:
    """
    Builds vocabulary: top-K words → indices.
    
    Selects top (VOCAB_SIZE - 1) words and assigns indices 1..VOCAB_SIZE-1.
    UNK token gets index 0.
    Final vocabulary size = VOCAB_SIZE (including UNK).
    """
    num_words_to_select = VOCAB_SIZE - 1  # Reserve index 0 for UNK
    print(f"\n  Building vocabulary (top {num_words_to_select:,} words + 1 UNK)...")
    
    sorted_words = sorted(total_freq.items(), key=lambda x: x[1], reverse=True)
    selected = sorted_words[:num_words_to_select]
    
    vocab = {UNK_TOKEN: 0}
    for idx, (word, _) in enumerate(selected, start=1):
        vocab[word] = idx
    
    print(f"  ✓ Created vocabulary: {len(vocab):,} total words (including {UNK_TOKEN})")
    return vocab

def save_vocabulary(
    vocab: Dict[str, int],
    total_freq: Dict[str, int],
    out_dir: str,
    decade_files,
    decade_frequencies: Dict[str, Dict[str, int]]
):
    """
    Saves vocabulary as JSON and generates statistics file.
    
    Args:
        vocab: word→index mapping
        total_freq: aggregated word frequencies
        out_dir: output directory for vocab.json
        decade_files: list of decade file paths
        decade_frequencies: pre-computed decade→{word→freq} (avoids re-reading files)
    """
    vocab_file = os.path.join(out_dir, "vocab.json")
    print(f"\n  Saving vocabulary to {vocab_file}...")
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved: {len(vocab):,} words")
    
    # Generate statistics file
    stats_file = vocab_file.replace('.json', '_stats.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("VOCABULARY - STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Vocabulary size: {len(vocab):,}\n")
        f.write(f"UNK token: {UNK_TOKEN} (index 0)\n")
        f.write(f"Top words indices: 1..{len(vocab)-1}\n\n")
        
        # Top 20 words
        f.write("Top 20 words:\n" + "-" * 50 + "\n")
        sorted_vocab = sorted(
            [(w, i) for w, i in vocab.items() if w != UNK_TOKEN],
            key=lambda x: x[1]
        )
        for word, idx in sorted_vocab[:20]:
            freq = total_freq.get(word, 0)
            f.write(f"{idx:6d}. {word:20s} freq={freq:12,d}\n")
        
        # Coverage per decade (reuse precomputed frequencies)
        f.write("\nCoverage per decade:\n" + "-" * 50 + "\n")
        for file_path in decade_files:
            decade = os.path.basename(file_path).replace('.txt', '')
            decade_freq = decade_frequencies.get(decade, {})
            present = sum(1 for w in vocab if w != UNK_TOKEN and w in decade_freq)
            coverage = 100 * present / (len(vocab) - 1) if len(vocab) > 1 else 0
            f.write(f"{decade}: {present:,}/{len(vocab)-1:,} ({coverage:.2f}%)\n")
    
    print(f"  ✓ Statistics saved to {stats_file}")


def main():
    """Build vocabulary from a specified processed directory."""
    parser = argparse.ArgumentParser(
        description="Build shared vocabulary from preprocessed decades",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build vocabulary from standard directory
  python src/data/build_vocab.py --processed-dir data/processed/5gram-full
  
  # Build vocabulary from custom path
  python src/data/build_vocab.py --processed-dir /path/to/processed
        """
    )
    
    parser.add_argument(
        '--processed-dir',
        type=str,
        required=True,
        help='Path to processed directory with decade files (required)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("BUILDING SHARED VOCABULARY")
    print("=" * 70)
    print(f"Target vocabulary size: {VOCAB_SIZE:,}")
    print(f"  (will select {VOCAB_SIZE-1:,} top words + 1 UNK = {VOCAB_SIZE:,} total)")
    
    # Validate directory exists
    processed_dir = os.path.abspath(args.processed_dir)
    if not os.path.isdir(processed_dir):
        print(f"❌ Directory not found: {processed_dir}")
        return 1
    
    # Get decade files
    decade_files = get_decade_files(processed_dir)
    if len(decade_files) < 3:
        print(f"❌ Not enough decade files in {processed_dir} (found {len(decade_files)}, need ≥3)")
        return 1
    
    print(f"\n✓ Using: {processed_dir}")
    print(f"  Found {len(decade_files)} decades: {[os.path.basename(f).replace('.txt', '') for f in decade_files[:3]]}{'...' if len(decade_files) > 3 else ''}")
    
    # Build vocabulary
    total_freq, decade_frequencies = compute_total_frequencies(decade_files)
    vocab = build_vocabulary(total_freq)
    save_vocabulary(vocab, total_freq, processed_dir, decade_files, decade_frequencies)
    
    print("\n" + "=" * 70)
    print("VOCABULARY BUILDING COMPLETED ✓")
    print("=" * 70)
    vocab_file = os.path.join(processed_dir, 'vocab.json')
    print(f"\nVocabulary file: {vocab_file}")
    print(f"Statistics file: {vocab_file.replace('.json', '_stats.txt')}")
    print("\nNext step: run training with src/models/train.py")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())