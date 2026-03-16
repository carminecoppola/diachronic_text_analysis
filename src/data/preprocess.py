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

Input: filtered n-gram file (format: ngram \t year \t count)
Output: per-decade files with cleaned n-grams (one per line, frequency-weighted)
"""

import os
import re
import unicodedata
from collections import defaultdict
from typing import List, Tuple, Optional
from tqdm import tqdm

from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, NGRAM_TYPE, START_YEAR, END_YEAR

# ============================================================================
# PARAMETRI CONFIGURABILI
# ============================================================================

MIN_TOKEN_LENGTH = 2              # Lunghezza minima token
MAX_TOKEN_LENGTH = 40             # Lunghezza massima token
MIN_ALPHA_RATIO = 0.6             # % minima caratteri alfabetici
MAX_DIGIT_COUNT = 3               # Massimo numero di cifre in un token
MAX_CHAR_REPETITION = 4           # Massimo caratteri ripetuti (aaaa → invalido)
MIN_OCCURRENCES = 5               # Soglia minima occorrenze per n-gram

# ============================================================================
# FUNZIONI DI PREPROCESSING
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalizza: lowercase + rimozione accenti Unicode."""
    text = text.lower()
    # Normalizzazione Unicode NFKD + rimozione caratteri combining
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
    Tokenizza e pulisce il testo secondo regole NLP per Word2Vec.
    
    Output: lista token puliti (stopword MANTENUTE, NO lemmatizzazione)
    """
    text = normalize_text(text)
    
    # Tokenizzazione: split su spazi/punteggiatura
    # Mantiene solo parole e numeri, scarta punteggiatura
    tokens = re.findall(r'\b[\w]+\b', text)
    
    cleaned = []
    for token in tokens:
        # Sostituisci numeri con NUM
        if is_number(token):
            cleaned.append('NUM')
        # Valida e aggiungi token linguistici
        elif is_valid_token(token):
            cleaned.append(token)
    
    return cleaned



def process_ngram_line(line: str) -> Optional[Tuple[List[str], int]]:
    """
    Processa una linea del file filtrato.
    
    Input:  "ngram TAB year TAB match_count TAB volume_count"
    Output: (tokens_puliti, year) o (None, None) se invalido
    """
    try:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            return None
        
        ngram_text = parts[0]
        year = int(parts[1])
        
        # Applica preprocessing completo
        tokens = tokenize_and_clean(ngram_text)
        
        # Scarta se dopo pulizia ha meno token del necessario
        if len(tokens) < NGRAM_TYPE:
            return None
        
        # Mantieni solo primi NGRAM_TYPE token per consistenza
        return tokens[:NGRAM_TYPE], year
        
    except (ValueError, IndexError):
        return None

def aggregate_by_decade(input_file: str, output_dir: str):
    """
    Aggrega n-gram per decennio applicando preprocessing NLP.
    
    Output: un file per decennio (es. 1900s.txt) contenente:
    - Un n-gram pulito per riga: "word1 word2 word3 word4 word5"
    - Replicato per numero di occorrenze (mantiene distribuzione naturale)
    """
    print(f"\n{'='*70}")
    print(f"PREPROCESSING NLP - Word2Vec/CBOW")
    print(f"{'='*70}")
    print(f"N-gram: {NGRAM_TYPE}, Anni: {START_YEAR}-{END_YEAR}, Min occ: {MIN_OCCURRENCES}")
    print(f"Parametri: len=[{MIN_TOKEN_LENGTH},{MAX_TOKEN_LENGTH}], "
          f"alpha≥{MIN_ALPHA_RATIO}, digit≤{MAX_DIGIT_COUNT}, rep≤{MAX_CHAR_REPETITION}")
    print(f"{'='*70}\n")
    
    # Aggregazione per decennio
    decade_data = defaultdict(lambda: defaultdict(int))
    
    print(f"Lettura e preprocessing: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processamento"):
            result = process_ngram_line(line)
            if result is None:
                continue
            tokens, year = result
            
            if tokens is None or year is None:
                continue
            
            if year < START_YEAR or year > END_YEAR:
                continue
            
            # Determina decennio (1995 → 1990)
            decade = (year // 10) * 10
            ngram_str = ' '.join(tokens)
            decade_data[decade][ngram_str] += 1
    
    print(f"\nDecenni trovati: {sorted(decade_data.keys())}\n")
    
    # Salvataggio
    os.makedirs(output_dir, exist_ok=True)
    
    for decade in sorted(decade_data.keys()):
        output_file = os.path.join(output_dir, f"{decade}s.txt")
        ngrams = decade_data[decade]
        
        # Filtra per occorrenze minime
        filtered = {ng: cnt for ng, cnt in ngrams.items() if cnt >= MIN_OCCURRENCES}
        
        print(f"{decade}s: {len(ngrams):,} totali → {len(filtered):,} filtrati (≥{MIN_OCCURRENCES})")
        
        # Scrivi n-gram replicati per frequenza
        with open(output_file, 'w', encoding='utf-8') as f:
            total = 0
            for ngram, count in filtered.items():
                for _ in range(count):
                    f.write(ngram + '\n')
                    total += 1
            print(f"  → {total:,} linee scritte in {output_file}")
    
    print(f"\n{'='*70}")
    print(f"PREPROCESSING COMPLETATO")
    print(f"{'='*70}\n")
    
    return True

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Entry point del preprocessing."""
    # Usa il file espanso come input e salva in una nuova directory
    input_file = os.path.join("data", "raw", "5gram_expanded", "5gram_filtered_expanded.tsv")
    output_dir = os.path.join("data", "processed", "5gram-full")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_file):
        print(f"File non trovato: {input_file}")
        return False
    
    success = aggregate_by_decade(input_file, output_dir)
    
    if success:
        print(f"✓ File salvati in: {output_dir}")
    else:
        print("Errore durante preprocessing")
    
    return success


if __name__ == "__main__":
    main()
