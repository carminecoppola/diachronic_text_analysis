#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute word frequencies per decade for alignment.

Reads preprocessed n-gram files from data/processed/ and computes token 
frequencies for each decade. Outputs frequency files in alignment/ directory.

These frequency files are required for temporal alignment algorithms
(e.g., Procrustes alignment, temporal referencing).

OPTIMIZATION: Uses streaming to avoid loading entire files in memory.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm

from src.config import ALIGNMENT_DIR

import glob
import os

def get_decade_files(processed_dir):
    """Restituisce lista di file decennali (1900s.txt, ...) in una directory."""
    files = sorted(glob.glob(os.path.join(processed_dir, "[0-9][0-9][0-9]0s.txt")))
    return [Path(f) for f in files]

# Output subdirectory for frequency files (organized by n-gram type)
def get_freq_output_dir(input_dir: Path) -> Path:
    return Path(ALIGNMENT_DIR) / input_dir.name / "frequencies"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def compute_frequencies_streaming(file_path: Path) -> Tuple[Counter, int, int]:
    """
    Compute token frequencies from n-gram file using streaming (memory-efficient).
    """
    freq = Counter()
    num_ngrams = 0
    total_tokens = 0
    
    logger.info(f"Processing: {file_path.name}")
    file_size = file_path.stat().st_size
    
    with open(file_path, 'r', encoding='utf-8') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Computing frequencies") as pbar:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    freq.update(tokens)
                    num_ngrams += 1
                    total_tokens += len(tokens)
                pbar.update(len(line.encode('utf-8')))
    return freq, num_ngrams, total_tokens

def print_statistics(decade: str, num_ngrams: int, total_tokens: int, freq: Counter):
    logger.info(f"Statistics for {decade}")
    logger.info(f"Unique tokens:           {len(freq):,}")
    logger.info(f"Total tokens:            {total_tokens:,}")
    logger.info(f"N-grams processed:       {num_ngrams:,}")
    if len(freq) > 0:
        most_common = freq.most_common(5)
        logger.info("Top 5 tokens:            " + ", ".join([f"{w} ({c})" for w, c in most_common]))

def save_frequencies_json(decade: str, freq: Counter, output_dir: Path):
    output_file = output_dir / f"freq_{decade}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(freq, f, ensure_ascii=False, indent=2)
    logger.info(f"✓ Saved JSON: {output_file.name}")

def save_frequencies_txt(decade: str, freq: Counter, output_dir: Path):
    output_file = output_dir / f"freq_{decade}.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, count in freq.most_common():
            f.write(f"{word}\t{count}\n")
    logger.info(f"✓ Saved TXT:  {output_file.name}")

def process_decade(file_path: Path, output_dir: Path) -> Tuple[str, Counter]:
    """
    Process a single decade file: compute frequencies and save results.
    """
    decade = file_path.stem
    logger.info(f"\n{'#'*70}")
    logger.info(f"Processing {decade}")
    logger.info(f"{'#'*70}")
    freq, num_ngrams, total_tokens = compute_frequencies_streaming(file_path)
    if not freq:
        logger.warning(f"No frequencies computed for {decade}")
        return decade, freq
    print_statistics(decade, num_ngrams, total_tokens, freq)
    logger.info(f"\nSaving frequency files...")
    save_frequencies_json(decade, freq, output_dir)
    save_frequencies_txt(decade, freq, output_dir)
    return decade, freq

def main(decades_filter: Optional[List[str]] = None):
    """
    Main execution function.
    """
    # Interattivo: scegli la directory di input
    base_dir = Path("data/processed")
    candidates = []
    for d in base_dir.iterdir():
        if d.is_dir():
            files = get_decade_files(d)
            if len(files) >= 3:
                candidates.append((d.name, d, files))
    if not candidates:
        logger.error("Nessuna directory con dati decennali trovata in data/processed/")
        sys.exit(1)
    print("\nOpzioni disponibili:")
    for i, (dname, dpath, files) in enumerate(candidates):
        print(f"  [{i+1}] {dname} ({len(files)} decenni)")
    scelta = input(f"\nScegli la directory [1-{len(candidates)}]: ").strip()
    try:
        idx = int(scelta) - 1
        assert 0 <= idx < len(candidates)
    except Exception:
        print("Scelta non valida.")
        sys.exit(1)
    dname, data_dir, decade_files = candidates[idx]
    output_dir = get_freq_output_dir(data_dir)
    print(f"\nUserò: {data_dir} → Output: {output_dir}")

    # Filtro decenni opzionale
    if decades_filter:
        decade_files = [f for f in decade_files if f.stem in decades_filter]
    if not decade_files:
        logger.error(f"No decade files found in {data_dir}")
        if decades_filter:
            logger.error(f"Filter applied: {decades_filter}")
        sys.exit(1)

    logger.info(f"{'='*70}")
    logger.info(f"Computing Word Frequencies for Alignment")
    logger.info(f"{'='*70}")
    logger.info(f"Input directory:  {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Decades found:    {len(decade_files)}")
    if decades_filter:
        logger.info(f"Filter applied:   {decades_filter}")
    logger.info(f"{'='*70}\n")

    freq_by_decade = {}
    for file_path in decade_files:
        decade, freq = process_decade(file_path, output_dir)
        freq_by_decade[decade] = freq

    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Successfully processed {len(freq_by_decade)} decades")
    logger.info(f"Decades: {sorted(freq_by_decade.keys())}")
    logger.info(f"\nFrequency files saved to: {output_dir}")
    logger.info(f"  - JSON format: freq_<decade>.json")
    logger.info(f"  - TXT format:  freq_<decade>.txt (tab-separated)")
    logger.info(f"\nDirectory structure:")
    logger.info(f"  {output_dir.parent.parent.name}/")
    logger.info(f"    └── {output_dir.parent.name}/")
    logger.info(f"         └── {output_dir.name}/")
    logger.info(f"\nThese files can now be used for temporal alignment!")
    logger.info(f"{'='*70}\n")
    return freq_by_decade

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute word frequencies per decade for temporal alignment"
    )
    parser.add_argument(
        '--decades', 
        nargs='+', 
        metavar='DECADE',
        help='Specific decades to process (e.g., 1980s 1990s)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress info messages (errors only)'
    )
    args = parser.parse_args()
    if args.quiet:
        logger.setLevel(logging.ERROR)
    freq_by_decade = main(decades_filter=args.decades)