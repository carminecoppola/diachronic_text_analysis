"""
Download parallelo di Google Books 5-gram con distribuzione alfabetica.
Scarica 20 file per ogni lettera dell'alfabeto usando download paralleli.
"""

import os
import sys
import gzip
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

# Aggiungi src al path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import (
    START_YEAR, END_YEAR, LANGUAGE, DATASET_VERSION,
    NGRAMS_BASE_URL, NGRAM_TYPE, ONLY_ALPHABETIC
)

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_RAW_EXPANDED_DIR = os.path.join(DATA_DIR, "raw", f"{NGRAM_TYPE}gram_expanded")
os.makedirs(DATA_RAW_EXPANDED_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(DATA_RAW_EXPANDED_DIR, f"{NGRAM_TYPE}gram_filtered_expanded.tsv")

# Mappatura lettere -> range file (basata su ricerca precedente)
LETTER_RANGES = {
    'a': (1350, 1370),
    'b': (1465, 1485),
    'c': (1553, 1573),
    'd': (1700, 1720),
    'e': (1766, 1786),
    'f': (1835, 1855),
    'g': (1900, 1920),
    'h': (2000, 2020),
    'i': (2075, 2095),
    'j': (2150, 2170),
    'k': (2200, 2220),
    'l': (2257, 2277),
    'm': (2325, 2345),
    'n': (2400, 2420),
    'o': (2450, 2470),
    'p': (2525, 2545),
    'q': (2575, 2595),
    'r': (2600, 2620),
    's': (2675, 2695),
    't': (2825, 2845),
    'u': (2975, 2995),
    'v': (3025, 3045),
    'w': (3062, 3082),
    'x': (3090, 3110),
    'y': (3100, 3120),
    'z': (3115, 3135),
}

# Parametri download parallelo
MAX_WORKERS = 8  # Download simultanei
TIMEOUT = 300    # Timeout per file (5 minuti)

# Lock per scrittura file output
output_lock = Lock()
stats_lock = Lock()

# Statistiche globali
global_stats = {
    'files_downloaded': 0,
    'files_exists': 0,
    'files_failed': 0,
    'lines_processed': 0,
    'lines_kept': 0
}

# ============================================================================
# FUNZIONI DOWNLOAD E FILTRAGGIO
# ============================================================================

def download_file(file_idx):
    """Scarica un singolo file se non esiste già."""
    filename = f"{NGRAM_TYPE}-{file_idx:05d}-of-19423.gz"
    filepath = os.path.join(DATA_RAW_EXPANDED_DIR, filename)
    
    if os.path.exists(filepath):
        return filepath, 'exists'
    
    url = f"{NGRAMS_BASE_URL}/{filename}"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            with open(filepath, 'wb') as f:
                f.write(response.read())
        return filepath, 'downloaded'
    except Exception as e:
        return None, f'error: {e}'


def filter_and_write_file(filepath, output_handle):
    """Filtra un file e scrive le righe valide nel file output."""
    valid_years = set(range(START_YEAR, END_YEAR + 1))
    lines_processed = 0
    lines_kept = 0
    
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as gz_f:
            for line in gz_f:
                lines_processed += 1
                
                try:
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue
                    
                    ngram = parts[0]
                    
                    # Filtra n-gram solo alfabetici
                    if ONLY_ALPHABETIC:
                        words = ngram.split()
                        if len(words) != NGRAM_TYPE:
                            continue
                        
                        # Pulisci parole
                        clean_words = []
                        for word in words:
                            if '_' in word:
                                word = word.split('_')[0]
                            word = word.strip('"\'.,;:!?-()[] ')
                            if word and word.isalpha():
                                clean_words.append(word.lower())
                        
                        if len(clean_words) != NGRAM_TYPE:
                            continue
                        
                        ngram = ' '.join(clean_words)
                    
                    # Processa entry anno,count,volume
                    for entry in parts[1:]:
                        if not entry.strip():
                            continue
                        
                        entry_parts = entry.split(',')
                        if len(entry_parts) != 3:
                            continue
                        
                        year_str, match_count, volume_count = entry_parts
                        year = int(year_str)
                        
                        if year in valid_years:
                            with output_lock:
                                output_handle.write(f"{ngram}\t{year}\t{match_count}\t{volume_count}\n")
                            lines_kept += 1
                
                except (ValueError, IndexError):
                    continue
        
        return lines_processed, lines_kept
    
    except Exception as e:
        print(f"\n⚠ Errore elaborazione {os.path.basename(filepath)}: {e}")
        return lines_processed, lines_kept


def process_file(file_idx, output_handle, pbar):
    """Processa un singolo file: download + filtraggio."""
    # Download
    filepath, status = download_file(file_idx)
    
    with stats_lock:
        if status == 'downloaded':
            global_stats['files_downloaded'] += 1
        elif status == 'exists':
            global_stats['files_exists'] += 1
        else:
            global_stats['files_failed'] += 1
            pbar.update(1)
            return False
    
    # Filtraggio
    if filepath:
        lines_proc, lines_kept = filter_and_write_file(filepath, output_handle)
        
        with stats_lock:
            global_stats['lines_processed'] += lines_proc
            global_stats['lines_kept'] += lines_kept
    
    pbar.update(1)
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Download parallelo con filtraggio."""
    print("="*70)
    print("DOWNLOAD PARALLELO GOOGLE BOOKS 5-GRAM")
    print("="*70)
    print(f"Workers paralleli: {MAX_WORKERS}")
    print(f"File per lettera: 20")
    print(f"Lettere: 26")
    print(f"File totali: {26 * 20}")
    print(f"Directory output: {DATA_RAW_EXPANDED_DIR}")
    print("="*70 + "\n")
    
    # Prepara lista di file da scaricare
    files_to_download = []
    for letter, (start, end) in sorted(LETTER_RANGES.items()):
        for file_idx in range(start, end):
            files_to_download.append(file_idx)
    
    total_files = len(files_to_download)
    print(f"File da processare: {total_files}\n")
    
    # Apri file output
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as output_f:
        
        # Progress bar
        with tqdm(total=total_files, desc="Download & Filtraggio", unit="file") as pbar:
            
            # Download parallelo
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [
                    executor.submit(process_file, file_idx, output_f, pbar)
                    for file_idx in files_to_download
                ]
                
                # Aspetta completamento
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"\n⚠ Errore: {e}")
    
    # Statistiche finali
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETATO")
    print("="*70)
    print(f"File scaricati: {global_stats['files_downloaded']}")
    print(f"File già esistenti: {global_stats['files_exists']}")
    print(f"File falliti: {global_stats['files_failed']}")
    print(f"Righe processate: {global_stats['lines_processed']:,}")
    print(f"Righe mantenute: {global_stats['lines_kept']:,}")
    if global_stats['lines_processed'] > 0:
        pct = 100 * global_stats['lines_kept'] / global_stats['lines_processed']
        print(f"Percentuale: {pct:.2f}%")
    print(f"Output: {OUTPUT_FILE}")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
