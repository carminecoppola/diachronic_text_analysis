#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP Preprocessing for Word2Vec/CBOW - Diachronic Text Analysis

Implements linguistic preprocessing tailored for Word2Vec:
1. Normalization: lowercase + Unicode accent removal
2. Tokenization: surface-level, no artificial boundaries (<s>, </s>)
3. Filtering: removes URLs, hashes, non-linguistic patterns  
4. Numbers: replaced with NUM token (preserves syntactic role)
5. Stopwords: RETAINED (essential for contextual stability)
6. Lemmatization: NOT applied (word forms have distinct distributions)

Input: filtered n-gram file (format: ngram \t year \t match_count \t volume_count)
Output: per-decade files with cleaned n-grams (one per line, frequency-weighted by match_count)
"""

import os
import re
import unicodedata
import argparse
from collections import defaultdict
from typing import List, Tuple, Optional
from tqdm import tqdm

from src.config import (
    NGRAM_TYPE, START_YEAR, END_YEAR, 
    get_preprocess_input_file, get_preprocess_output_dir, get_decade_from_year
)

# ============================================================================
# CONFIGURABLE PARAMETERS
# ============================================================================

MIN_TOKEN_LENGTH = 2              # Minimum token length
MAX_TOKEN_LENGTH = 40             # Maximum token length
MIN_ALPHA_RATIO = 0.6             # Minimum percentage of alphabetic characters
MAX_DIGIT_COUNT = 3               # Maximum number of digits in a token
MAX_CHAR_REPETITION = 4           # Maximum repeated characters (aaaa → invalid)
MIN_OCCURRENCES = 5               # Minimum occurrence threshold for n-grams

# Precompiled regex for tokenization (more efficient)
TOKENIZE_REGEX = re.compile(r'\b[\w]+\b')

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalizes: lowercase + Unicode accent removal."""
    text = text.lower()
    # Unicode NFKD normalization + removal of combining characters
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in nfkd if not unicodedata.combining(c)])


def is_number(token: str) -> bool:
    """
    True se token è numero puro/data (123, 1990, 1,000).
    False per alfanumerici con valore lessicale (mp3, covid19).
    """
    clean = token.replace(',', '').replace('.', '')
    return clean.isdigit()


def is_repetitive(token: str) -> bool:
    """True se token ha ripetizioni artificiali (aaaa, !!!!)."""
    if len(token) < MAX_CHAR_REPETITION:
        return False
    for i in range(len(token) - MAX_CHAR_REPETITION + 1):
        if len(set(token[i:i+MAX_CHAR_REPETITION])) == 1:
            return True
    return False

def is_valid_token(token: str) -> bool:
    """
    Verifica validità token. Scartato se soddisfa ALMENO UNA condizione:
    - Lunghezza fuori range [MIN_TOKEN_LENGTH, MAX_TOKEN_LENGTH]
    - % caratteri alfabetici < MIN_ALPHA_RATIO
    - URL/hash/codici (http, www, .com, #, @)
    - Ripetizioni artificiali (aaaa)
    - Troppi numeri (>MAX_DIGIT_COUNT e <50% alfabetici)
    
    NON scarta token solo perché rari (frequenza controllata dopo).
    """
    # Controllo lunghezza
    if len(token) < MIN_TOKEN_LENGTH or len(token) > MAX_TOKEN_LENGTH:
        return False
    
    # Pattern non linguistici
    if any(p in token for p in ['http', 'www', '.com', '#', '@']):
        return False
    
    # % caratteri alfabetici
    alpha_count = sum(1 for c in token if c.isalpha())
    alpha_ratio = alpha_count / len(token)
    if alpha_ratio < MIN_ALPHA_RATIO:
        return False
    
    # Deve contenere almeno una lettera
    if alpha_count == 0:
        return False
    
    # Ripetizioni artificiali
    if is_repetitive(token):
        return False
    
    # Troppi numeri (ma permetti alfanumerici come mp3)
    digit_count = sum(1 for c in token if c.isdigit())
    if digit_count > MAX_DIGIT_COUNT and alpha_ratio < 0.5:
        return False
    
    return True


def tokenize_and_clean(text: str) -> List[str]:
    """
    Tokenizes and cleans text according to NLP rules for Word2Vec.
    
    Output: list of cleaned tokens (stopwords RETAINED, no lemmatization)
    """
    text = normalize_text(text)
    
    # Tokenization: split on whitespace/punctuation
    # Keeps only words and numbers, discards punctuation
    tokens = TOKENIZE_REGEX.findall(text)
    
    cleaned = []
    for token in tokens:
        # Replace numbers with NUM token
        if is_number(token):
            cleaned.append('NUM')
        # Validate and add linguistic tokens
        elif is_valid_token(token):
            cleaned.append(token)
    
    return cleaned



def process_ngram_line(line: str) -> Optional[Tuple[List[str], int, int]]:
    """
    Processes a line from the filtered n-gram file.
    
    Input:  "ngram TAB year TAB match_count TAB volume_count"
    Output: (tokens_cleaned, year, match_count) or None if invalid
    
    The match_count field represents how many times this n-gram appeared
    in the corpus for the given year - used for accurate frequency weighting.
    """
    try:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            return None
        
        ngram_text = parts[0]
        year = int(parts[1])
        match_count = int(parts[2])
        
        # Apply complete preprocessing
        tokens = tokenize_and_clean(ngram_text)
        
        # Discard if after cleaning has fewer tokens than n-gram size
        if len(tokens) < NGRAM_TYPE:
            return None
        
        # Keep only first NGRAM_TYPE tokens for consistency
        return tokens[:NGRAM_TYPE], year, match_count
        
    except (ValueError, IndexError):
        return None

def aggregate_by_decade(input_file: str, output_dir: str):
    """
    Aggregates n-grams by decade applying NLP preprocessing.
    
    Uses match_count field to weight frequencies accurately:
    - match_count represents actual corpus occurrences per year
    - Total frequency = sum of match_counts across years in the decade
    
    Output: one file per decade (e.g. 1900s.txt) containing:
    - One cleaned n-gram per line: "word1 word2 word3 word4 word5"
    - Replicated by match_count to preserve natural distribution
    """
    print(f"\n{'='*70}")
    print(f"NLP PREPROCESSING - Word2Vec/CBOW")
    print(f"{'='*70}")
    print(f"N-gram: {NGRAM_TYPE}, Years: {START_YEAR}-{END_YEAR}, Min occ: {MIN_OCCURRENCES}")
    print(f"Parameters: len=[{MIN_TOKEN_LENGTH},{MAX_TOKEN_LENGTH}], "
          f"alpha≥{MIN_ALPHA_RATIO}, digit≤{MAX_DIGIT_COUNT}, rep≤{MAX_CHAR_REPETITION}")
    print(f"{'='*70}\n")
    
    # Aggregation by decade
    decade_data = defaultdict(lambda: defaultdict(int))
    
    print(f"Reading and preprocessing: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing"):
            result = process_ngram_line(line)
            if result is None:
                continue
            
            tokens, year, match_count = result
            
            if year < START_YEAR or year > END_YEAR:
                continue
            
            # Determine decade (1995 → 1990s)
            decade_str = get_decade_from_year(year)
            decade_int = int(decade_str[:-1])  # Convert "1990s" → 1990
            ngram_str = ' '.join(tokens)
            
            # Accumulate match_count (not just increment by 1)
            decade_data[decade_int][ngram_str] += match_count
    
    print(f"\nDecades found: {sorted(decade_data.keys())}\n")
    
    # Save to files
    os.makedirs(output_dir, exist_ok=True)
    
    for decade in sorted(decade_data.keys()):
        output_file = os.path.join(output_dir, f"{decade}s.txt")
        ngrams = decade_data[decade]
        
        # Filter by minimum occurrences
        filtered = {ng: cnt for ng, cnt in ngrams.items() if cnt >= MIN_OCCURRENCES}
        
        print(f"{decade}s: {len(ngrams):,} total → {len(filtered):,} filtered (≥{MIN_OCCURRENCES})")
        
        # Write n-grams replicated by frequency
        with open(output_file, 'w', encoding='utf-8') as f:
            total = 0
            for ngram, count in filtered.items():
                for _ in range(count):
                    f.write(ngram + '\n')
                    total += 1
            print(f"  → {total:,} lines written to {output_file}")
    
    print(f"\n{'='*70}")
    print(f"PREPROCESSING COMPLETED")
    print(f"{'='*70}\n")
    
    return True

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Entry point for preprocessing with configurable parameters."""
    parser = argparse.ArgumentParser(
        description="NLP preprocessing for Word2Vec - Diachronic Text Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default expanded dataset
  python src/data/preprocess.py
  
  # Use standard dataset, custom output
  python src/data/preprocess.py --expanded=False --output-dir /tmp/output
  
  # Specify custom input and output
  python src/data/preprocess.py --input-file /path/to/ngrams.tsv --output-dir /path/to/output
        """
    )
    
    parser.add_argument(
        '--input-file',
        type=str,
        default=None,
        help='Path to input n-gram file (TSV format). Defaults to expanded dataset path from config.'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save processed files. Defaults to processed dataset path from config.'
    )
    
    parser.add_argument(
        '--expanded',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help='Use expanded dataset (True) or standard dataset (False). Default: True'
    )
    
    args = parser.parse_args()
    
    # Determine input and output paths
    if args.input_file is None:
        # Use default from config based on --expanded flag
        input_file = get_preprocess_input_file(expanded=args.expanded)
    else:
        input_file = args.input_file
    
    if args.output_dir is None:
        # Use default from config
        output_dir = get_preprocess_output_dir(expanded=args.expanded)
    else:
        output_dir = args.output_dir
    
    print(f"\nConfiguration:")
    print(f"  Input file:  {input_file}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Expanded:    {args.expanded}")
    
    if not os.path.exists(input_file):
        print(f"\n✗ Input file not found: {input_file}")
        return False
    
    success = aggregate_by_decade(input_file, output_dir)
    
    if success:
        print(f"✓ Files saved to: {output_dir}")
    else:
        print("✗ Error during preprocessing")
    
    return success


if __name__ == "__main__":
    main()
