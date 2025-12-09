"""
Script per Download e Organizzazione Google Books N-grams

Questo script scarica e organizza i dati Google Books N-grams 1-gram
per l'analisi diacronica. I dati vengono filtrati per periodi temporali
specifici e salvati in formato semplificato.

Google Books N-grams formato:
    ngram TAB year TAB match_count TAB volume_count

Dataset: http://storage.googleapis.com/books/ngrams/books/datasetsv3.html
"""

import os
import sys
import gzip
import urllib.request
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import config


# URL base per Google Books N-grams (versione 3, inglese)
NGRAMS_BASE_URL = "http://storage.googleapis.com/books/ngrams/books/20200217/eng/"

# File 1-gram disponibili (A-Z, altri caratteri)
# Per demo/testing useremo solo alcune lettere
NGRAM_FILES = [
    "1-00000-of-00024.gz",  # contiene parole comuni
    "1-00001-of-00024.gz",
    "1-00002-of-00024.gz",
]

# Per progetto completo, decommentare per scaricare tutto:
# NGRAM_FILES = [f"1-{i:05d}-of-00024.gz" for i in range(24)]


def download_ngram_file(filename, output_dir):
    """
    Scarica un file N-gram da Google Books.
    
    Args:
        filename: Nome del file da scaricare
        output_dir: Directory di destinazione
    
    Returns:
        Path del file scaricato
    """
    url = NGRAMS_BASE_URL + filename
    output_path = output_dir / filename
    
    if output_path.exists():
        print(f"  ✓ File già esistente: {filename}")
        return output_path
    
    print(f"  Downloading: {filename}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"  ✓ Completato: {filename}")
        return output_path
    except Exception as e:
        print(f"  ✗ Errore download {filename}: {e}")
        return None


def process_ngram_file(filepath, period_data, min_year, max_year):
    """
    Processa un file N-gram e aggrega dati per periodo.
    
    Args:
        filepath: Path del file .gz da processare
        period_data: Dict per accumulare dati {period: {word: count}}
        min_year: Anno minimo da considerare
        max_year: Anno massimo da considerare
    """
    print(f"  Processing: {filepath.name}...")
    
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    parts = line.strip().split('\t')
                    if len(parts) != 4:
                        continue
                    
                    ngram, year, match_count, volume_count = parts
                    year = int(year)
                    match_count = int(match_count)
                    
                    # Filtra per range temporale
                    if year < min_year or year > max_year:
                        continue
                    
                    # Filtra parole con caratteri non alfabetici
                    if not ngram.isalpha():
                        continue
                    
                    # Determina periodo decennale
                    decade = (year // 10) * 10
                    period = f"{decade}s"
                    
                    # Accumula conteggi
                    if period not in period_data:
                        period_data[period] = defaultdict(int)
                    
                    period_data[period][ngram.lower()] += match_count
                    
                except (ValueError, IndexError):
                    continue
    
    except Exception as e:
        print(f"  ✗ Errore processing {filepath.name}: {e}")


def save_period_data(period, word_counts, output_dir, top_n=100000):
    """
    Salva i dati di un periodo in formato semplificato.
    
    Args:
        period: Nome del periodo (es. '1900s')
        word_counts: Dict {word: count}
        output_dir: Directory di output
        top_n: Numero di parole più frequenti da salvare
    """
    # Ordina per frequenza e prendi top N
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    output_file = output_dir / f"{period}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, count in sorted_words:
            # Ripeti la parola in base alla frequenza (sampling approssimativo)
            # Per corpus reale, questo crea un file di testo utilizzabile
            # Normalizziamo: 1 occorrenza ogni 1000 match_count
            repetitions = max(1, count // 1000)
            f.write(f"{word} " * repetitions)
            if repetitions > 10:  # Aggiungi newline ogni tanto
                f.write("\n")
    
    print(f"  ✓ Salvato {period}: {len(sorted_words)} parole uniche -> {output_file}")


def main():
    """
    Main function: scarica e organizza Google Books N-grams.
    """
    print("="*70)
    print("  Google Books N-grams - Download e Organizzazione")
    print("="*70)
    print()
    
    # Crea directory temporanea per download
    download_dir = config.DATA_DIR / "ngrams_raw"
    download_dir.mkdir(exist_ok=True)
    
    # Range temporale per analisi
    MIN_YEAR = 1900
    MAX_YEAR = 1999
    
    print(f"Range temporale: {MIN_YEAR}-{MAX_YEAR}")
    print(f"Periodi: {config.PERIODS}")
    print()
    
    # Step 1: Download files
    print("[1/3] Download N-gram files...")
    downloaded_files = []
    for filename in NGRAM_FILES:
        filepath = download_ngram_file(filename, download_dir)
        if filepath:
            downloaded_files.append(filepath)
    print()
    
    if not downloaded_files:
        print("✗ Nessun file scaricato. Verifica connessione.")
        return
    
    # Step 2: Process files e aggrega per periodo
    print("[2/3] Processing e aggregazione per periodo...")
    period_data = {}
    
    for filepath in downloaded_files:
        process_ngram_file(filepath, period_data, MIN_YEAR, MAX_YEAR)
    print()
    
    # Step 3: Salva dati per periodo
    print("[3/3] Salvataggio dati per periodo...")
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for period in sorted(period_data.keys()):
        save_period_data(period, period_data[period], config.RAW_DATA_DIR)
    print()
    
    # Statistiche finali
    print("="*70)
    print("  ✅ COMPLETATO!")
    print("="*70)
    print(f"Periodi processati: {len(period_data)}")
    for period in sorted(period_data.keys()):
        print(f"  - {period}: {len(period_data[period])} parole uniche")
    print()
    print(f"Dati salvati in: {config.RAW_DATA_DIR}")
    print()
    
    # Cleanup opzionale
    cleanup = input("Rimuovere file .gz scaricati? (y/n): ")
    if cleanup.lower() == 'y':
        for filepath in downloaded_files:
            filepath.unlink()
        download_dir.rmdir()
        print("✓ File temporanei rimossi")


if __name__ == "__main__":
    main()
