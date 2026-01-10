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
from typing import Tuple, Dict
from tqdm import tqdm

from src.config import DATA_PROCESSED_DIR, ALIGNMENT_DIR, NGRAM_SUFFIX


# Output subdirectory for frequency files (organized by n-gram type)
FREQ_OUTPUT_DIR = Path(ALIGNMENT_DIR) / NGRAM_SUFFIX / "frequencies"

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
    
    OPTIMIZATION: Instead of loading all n-grams into memory first, this function
    processes the file line-by-line and updates the Counter incrementally.
    This reduces memory usage from O(n) to O(vocab_size), where n = total tokens.
    
    Args:
        file_path: Path to text file (one n-gram per line)
    
    Returns:
        Tuple of (freq_counter, num_ngrams, total_tokens)
    """
    freq = Counter()
    num_ngrams = 0
    total_tokens = 0
    
    logger.info(f"Processing: {file_path.name}")
    
    # IMPROVEMENT: Get file size for progress bar (shows MB instead of line count)
    file_size = file_path.stat().st_size
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # IMPROVEMENT: Progress bar based on bytes read, not lines
        with tqdm(total=file_size, unit='B', unit_scale=True, 
                  desc=f"Computing frequencies") as pbar:
            for line in f:
                tokens = line.strip().split()
                if tokens:  # Skip empty lines
                    # OPTIMIZATION: Counter.update() is more efficient than looping
                    # and incrementing individual tokens
                    freq.update(tokens)
                    num_ngrams += 1
                    total_tokens += len(tokens)
                
                # Update progress bar based on bytes read
                pbar.update(len(line.encode('utf-8')))
    
    # IMPROVEMENT: Return metrics calculated during streaming (no need to recalculate)
    return freq, num_ngrams, total_tokens


def print_statistics(decade: str, num_ngrams: int, total_tokens: int, freq: Counter):
    """
    Print statistics for computed frequencies.
    
    Args:
        decade: Decade name (e.g., "1980s")
        num_ngrams: Number of n-grams processed
        total_tokens: Total number of tokens
        freq: Counter object with token frequencies
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Statistics for {decade}")
    logger.info(f"{'='*70}")
    logger.info(f"N-grams processed:       {num_ngrams:,}")
    logger.info(f"Total tokens:            {total_tokens:,}")
    logger.info(f"Unique tokens (vocab):   {len(freq):,}")
    logger.info(f"Average tokens per gram: {total_tokens/num_ngrams:.2f}")
    logger.info(f"\nTop-20 most frequent words:")
    for i, (word, count) in enumerate(freq.most_common(20), 1):
        logger.info(f"  {i:2d}. {word:20s} : {count:,}")


def save_frequencies_json(decade: str, freq: Counter, output_dir: Path):
    """
    Save frequency dictionary to JSON file.
    
    Args:
        decade: Decade name (e.g., "1980s")
        freq: Counter object with token frequencies
        output_dir: Directory to save JSON files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"freq_{decade}.json"
    
    # Convert Counter to regular dict for JSON serialization
    freq_dict = dict(freq)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(freq_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Saved JSON: {output_file.name}")


def save_frequencies_txt(decade: str, freq: Counter, output_dir: Path):
    """
    Save frequency dictionary to plain text file (tab-separated).
    Format: word\tcount
    
    This format is often required by alignment tools.
    
    Args:
        decade: Decade name (e.g., "1980s")
        freq: Counter object with token frequencies
        output_dir: Directory to save text files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"freq_{decade}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, count in freq.most_common():
            f.write(f"{word}\t{count}\n")
    
    logger.info(f"✓ Saved TXT:  {output_file.name}")


def process_decade(file_path: Path, output_dir: Path) -> Tuple[str, Counter]:
    """
    Process a single decade file: compute frequencies and save results.
    
    REFACTORING: Extracted decade processing logic into separate function
    for better modularity and testability.
    
    Args:
        file_path: Path to decade file
        output_dir: Output directory for frequency files
    
    Returns:
        Tuple of (decade_name, freq_counter)
    """
    decade = file_path.stem
    
    logger.info(f"\n{'#'*70}")
    logger.info(f"Processing {decade}")
    logger.info(f"{'#'*70}")
    
    # CHANGE: Use streaming function instead of load_ngrams() + compute_frequencies()
    freq, num_ngrams, total_tokens = compute_frequencies_streaming(file_path)
    
    if not freq:
        logger.warning(f"No frequencies computed for {decade}")
        return decade, freq
    
    # Print statistics
    print_statistics(decade, num_ngrams, total_tokens, freq)
    
    # Save to JSON and TXT formats
    logger.info(f"\nSaving frequency files...")
    save_frequencies_json(decade, freq, output_dir)
    save_frequencies_txt(decade, freq, output_dir)
    
    return decade, freq


def main(decades_filter: list = None):
    """
    Main execution function.
    
    IMPROVEMENT: Added optional filtering and better error handling.
    
    Args:
        decades_filter: Optional list of specific decades to process (e.g., ["1980s", "1990s"])
    """
    # Get all decade files
    data_dir = Path(DATA_PROCESSED_DIR)
    output_dir = FREQ_OUTPUT_DIR
    
    # IMPROVEMENT: Validate input directory exists before processing
    if not data_dir.exists():
        logger.error(f"Input directory not found: {data_dir}")
        sys.exit(1)
    
    decade_files = sorted(data_dir.glob("*.txt"))
    # Exclude non-decade files (vocab.txt, vocab_stats.txt, etc.)
    decade_files = [f for f in decade_files if f.stem.endswith('s') and f.stem[:-1].isdigit()]
    
    # FEATURE: Apply filter if specified (allows processing specific decades only)
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
    logger.info(f"N-gram type:      {NGRAM_SUFFIX}")
    logger.info(f"Decades found:    {len(decade_files)}")
    if decades_filter:
        logger.info(f"Filter applied:   {decades_filter}")
    logger.info(f"{'='*70}\n")
    
    freq_by_decade = {}
    
    # Process each decade
    for file_path in decade_files:
        decade, freq = process_decade(file_path, output_dir)
        freq_by_decade[decade] = freq
    
    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Successfully processed {len(freq_by_decade)} decades")
    logger.info(f"Decades: {sorted(freq_by_decade.keys())}")
    logger.info(f"\nFrequency files saved to: {output_dir}")
    logger.info(f"  - JSON format: freq_<decade>.json")
    logger.info(f"  - TXT format:  freq_<decade>.txt (tab-separated)")
    logger.info(f"\nDirectory structure:")
    logger.info(f"  {Path(ALIGNMENT_DIR).name}/")
    logger.info(f"    └── {NGRAM_SUFFIX}/")
    logger.info(f"         └── {output_dir.name}/")
    logger.info(f"\nThese files can now be used for temporal alignment!")
    logger.info(f"{'='*70}\n")
    
    return freq_by_decade


if __name__ == "__main__":
    # FEATURE: Added command-line argument parsing for better usability
    parser = argparse.ArgumentParser(
        description="Compute word frequencies per decade for temporal alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all decades
  python compute_frequencies.py
  
  # Process specific decades
  python compute_frequencies.py --decades 1980s 1990s 2000s
  
  # Quiet mode (errors only)
  python compute_frequencies.py --quiet
        """
    )
    # FEATURE: Allow processing specific decades only
    parser.add_argument(
        '--decades', 
        nargs='+', 
        metavar='DECADE',
        help='Specific decades to process (e.g., 1980s 1990s)'
    )
    # FEATURE: Quiet mode for batch processing
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress info messages (errors only)'
    )
    
    args = parser.parse_args()
    
    # IMPROVEMENT: Adjust logging level based on command-line flags
    if args.quiet:
        logger.setLevel(logging.ERROR)
    
    freq_by_decade = main(decades_filter=args.decades)
