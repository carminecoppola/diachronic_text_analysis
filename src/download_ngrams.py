"""
FASE 1: Download e filtraggio Google Books N-grams

Questo script gestisce il download dei dati Google Books N-grams (1-grams)
e il filtraggio iniziale per anni di interesse.

COSA FA:
1. Scarica file 1-gram inglesi da Google Cloud Storage
2. Scarica file total_counts (necessario per normalizzazione)
3. Filtra righe mantenendo solo anni 1900-1990
4. Salva dati filtrati in data/raw/

NOTA IMPORTANTE:
I file completi sono MOLTO grandi (centinaia di GB).
Per scopi universitari/test, è sufficiente scaricare un subset (3-5 file).

OUTPUT:
- data/raw/1gram_filtered.tsv: N-gram filtrati per anni
- data/raw/total_counts.txt: Totali per normalizzazione

Formato output:
ngram \t year \t match_count \t volume_count
"""

import os
import sys
import gzip
import urllib.request
from tqdm import tqdm
from typing import Set

# Import config
from config import (
    DATA_RAW_DIR,
    TOTAL_COUNTS_FILE,
    START_YEAR,
    END_YEAR,
    LANGUAGE,
    DATASET_VERSION,
    NGRAMS_BASE_URL,
    NUM_FILES_TO_DOWNLOAD,
    ONLY_ALPHABETIC,
    create_directories
)


def download_file(url: str, dest_path: str) -> bool:
    """
    Scarica un file con progress bar.
    
    Args:
        url: URL del file da scaricare
        dest_path: Path di destinazione
    
    Returns:
        True se successo, False altrimenti
    """
    try:
        print(f"Downloading {url}...")
        
        # Request con user agent
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                chunk_size = 8192
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    
    except Exception as e:
        print(f"Errore download: {e}")
        return False


def download_total_counts() -> bool:
    """
    Scarica il file total_counts necessario per normalizzazione (V3 format).
    
    Returns:
        True se successo, False altrimenti
    """
    print("\n" + "="*70)
    print("DOWNLOAD TOTAL_COUNTS")
    print("="*70)
    
    if os.path.exists(TOTAL_COUNTS_FILE):
        print(f"✓ File già presente: {TOTAL_COUNTS_FILE}")
        return True
    
    # V3 2020 usa totalcounts-1 (non più il vecchio formato)
    url = f"{NGRAMS_BASE_URL}/totalcounts-1"
    
    print(f"Tentativo download da: {url}")
    if download_file(url, TOTAL_COUNTS_FILE):
        return True
    
    print("\n⚠️  Total_counts non disponibile per questa versione.")
    print("Il progetto può continuare senza normalizzazione (userà frequenze raw)")
    print("Creo file segnaposto...")
    
    # Crea file segnaposto
    with open(TOTAL_COUNTS_FILE, 'w') as f:
        f.write("# Total counts non disponibile per versione dataset\n")
    
    return True  # Continua comunque


def download_and_filter_ngrams() -> bool:
    """
    Scarica file 1-gram e filtra per anni di interesse.
    
    Returns:
        True se successo, False altrimenti
    """
    print("\n" + "="*70)
    print("DOWNLOAD E FILTRAGGIO 1-GRAMS")
    print("="*70)
    print(f"Periodo: {START_YEAR}-{END_YEAR}")
    print(f"File da scaricare: {NUM_FILES_TO_DOWNLOAD} (su 24 totali)")
    print(f"ATTENZIONE: Questo può richiedere molto tempo e spazio disco!")
    print("="*70 + "\n")
    
    # File output
    output_file = os.path.join(DATA_RAW_DIR, "1gram_filtered.tsv")
    
    # Set di anni validi
    valid_years = set(range(START_YEAR, END_YEAR + 1))
    
    total_lines_processed = 0
    total_lines_kept = 0
    
    # Apri file output
    with open(output_file, 'w', encoding='utf-8') as out_f:
        
        # Scarica e processa file 1-gram
        for file_idx in range(NUM_FILES_TO_DOWNLOAD):
            
            # V3 2020: formato 1-00000-of-00024.gz, 1-00001-of-00024.gz, etc.
            filename = f"1-{file_idx:05d}-of-00024.gz"
            url = f"{NGRAMS_BASE_URL}/{filename}"
            
            # Path temporaneo
            temp_file = os.path.join(DATA_RAW_DIR, filename)
            
            print(f"\n[{file_idx+1}/{NUM_FILES_TO_DOWNLOAD}] Elaborazione {filename}...")
            
            # Download
            if not os.path.exists(temp_file):
                success = download_file(url, temp_file)
                if not success:
                    print(f"⚠ Impossibile scaricare {filename}, salto...")
                    continue
            else:
                print(f"✓ File già scaricato: {temp_file}")
            
            # Filtraggio
            print("Filtraggio righe (formato V3 compatto)...")
            lines_processed = 0
            lines_kept = 0
            
            try:
                with gzip.open(temp_file, 'rt', encoding='utf-8') as gz_f:
                    for line in gz_f:
                        lines_processed += 1
                        
                        if lines_processed % 100000 == 0:
                            print(f"  Processate {lines_processed//1000}K righe, mantenute {lines_kept}...")
                        
                        try:
                            # Formato V3: ngram \t anno1,count1,vol1 \t anno2,count2,vol2 ...
                            # Esempio: "computer\t1950,42,12\t1951,89,23\t1952,156,34"
                            # Significa: "computer" appare 42 volte in 12 volumi nel 1950, ecc.
                            parts = line.strip().split('\t')
                            if len(parts) < 2:
                                continue
                            
                            ngram = parts[0]
                            
                            # Filtra solo alfabetici (se richiesto)
                            # Rimuove numeri, punteggiatura, simboli speciali
                            if ONLY_ALPHABETIC and not ngram.isalpha():
                                continue
                            
                            # Processa ogni entry anno,count,volume
                            # Ogni entry contiene: anno,match_count,volume_count
                            for entry in parts[1:]:
                                if not entry.strip():
                                    continue
                                
                                entry_parts = entry.split(',')
                                if len(entry_parts) != 3:
                                    continue
                                
                                year_str, match_count, volume_count = entry_parts
                                year = int(year_str)
                                
                                # Filtra anno (mantieni solo 1900-1990)
                                if year in valid_years:
                                    # Scrivi in formato esploso: ngram \t year \t match \t volume
                                    # Trasformazione: una riga compatta → molte righe (una per anno)
                                    # Esempio: "war\t1940\t125643\t5234" = "war" nel 1940 con 125643 occorrenze
                                    out_f.write(f"{ngram}\t{year}\t{match_count}\t{volume_count}\n")
                                    lines_kept += 1
                        
                        except (ValueError, IndexError):
                            # Riga malformata, skippa
                            continue
                
                print(f"✓ Completato: {lines_processed} righe processate, {lines_kept} mantenute")
                total_lines_processed += lines_processed
                total_lines_kept += lines_kept
                
                # Rimuovi file temporaneo per risparmiare spazio
                # (commentare se si vuole mantenere originale)
                # os.remove(temp_file)
            
            except Exception as e:
                print(f"⚠ Errore elaborazione {filename}: {e}")
                continue
    
    print("\n" + "="*70)
    print("FILTRAGGIO COMPLETATO")
    print("="*70)
    print(f"Righe totali processate: {total_lines_processed:,}")
    print(f"Righe mantenute: {total_lines_kept:,}")
    print(f"Percentuale: {100*total_lines_kept/max(total_lines_processed,1):.2f}%")
    print(f"Output salvato in: {output_file}")
    print("="*70)
    
    return True


def main():
    """Main function."""
    
    print("="*70)
    print("GOOGLE BOOKS N-GRAMS - DOWNLOAD E FILTRAGGIO")
    print("="*70)
    print(f"Dataset: Google Books N-grams v3 ({DATASET_VERSION})")
    print(f"Lingua: {LANGUAGE}")
    print(f"Periodo: {START_YEAR}-{END_YEAR}")
    print("="*70 + "\n")
    
    # Crea directory
    create_directories()
    
    # Step 1: Download total_counts
    if not download_total_counts():
        print("❌ Errore download total_counts. Impossibile continuare.")
        return 1
    
    # Step 2: Download e filtra 1-grams
    if not download_and_filter_ngrams():
        print("❌ Errore download/filtraggio N-grams.")
        return 1
    
    print("\n✅ FASE 1 COMPLETATA")
    print("Prossimo step: eseguire preprocess.py per aggregazione per decennio")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
