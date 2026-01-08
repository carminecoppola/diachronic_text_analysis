"""
FASE 2: Preprocessing e aggregazione per decennio

Questo script processa i dati filtrati da FASE 1 e li aggrega per decennio.

COSA FA:
1. Legge data/raw/{NGRAM_TYPE}gram_filtered.tsv (supporta 3-gram e 5-gram)
2. Aggrega frequenze per decennio (1900-1909 → 1900s, ecc.)
3. Normalizza frequenze usando total_counts
4. Applica filtri di pulizia (lunghezza, lowercase)
5. Salva file separati per ogni decennio in data/processed/{NGRAM_TYPE}gram/

INPUT:
- data/raw/{NGRAM_TYPE}gram/Ngram_filtered.tsv
- data/raw/{NGRAM_TYPE}gram/total_counts.txt

OUTPUT:
- data/processed/{NGRAM_TYPE}gram/1900s.txt
- data/processed/{NGRAM_TYPE}gram/1910s.txt
- ...
- data/processed/{NGRAM_TYPE}gram/2010s.txt

Formato output: ogni file contiene righe con:
ngram \t normalized_frequency

Dove ngram è nel formato:
- 3-gram: "word1 word2 word3" (3 parole separate da spazio)
- 5-gram: "word1 word2 word3 word4 word5" (5 parole separate da spazio)

Questo MANTIENE il contesto originale per l'analisi diacronica.

NOTA IMPORTANTE:
NON spacchetta gli n-gram in parole singole.
Il contesto delle sequenze viene preservato per il training CBOW.
"""

import os
import sys
from collections import defaultdict
from typing import Dict

# Import config
from config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    TOTAL_COUNTS_FILE,
    START_YEAR,
    END_YEAR,
    DECADES,
    NGRAM_TYPE,
    MIN_TOKEN_LENGTH,
    MAX_TOKEN_LENGTH,
    LOWERCASE,
    MIN_CORPUS_OCCURRENCES,
    get_decade_from_year,
    get_processed_file_path,
    create_directories
)


def load_total_counts() -> Dict[int, int]:
    """
    Carica total_counts per normalizzazione.
    
    Returns:
        Dizionario {anno: total_match_count}
    """
    print("Caricamento total_counts...")
    
    total_counts = {}
    
    try:
        with open(TOTAL_COUNTS_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # V3 format: una riga con valori separati da virgole
            # Formato: year,match_count,page_count,volume_count ripetuto
            parts = content.split(',')
            
            # Processa a gruppi di 4 (year, match_count, page_count, volume_count)
            for i in range(0, len(parts), 4):
                if i + 3 < len(parts):
                    try:
                        year = int(parts[i].strip())
                        match_count = int(parts[i+1].strip())
                        total_counts[year] = match_count
                    except ValueError:
                        continue
    
    except FileNotFoundError:
        print(f"⚠ File total_counts non trovato: {TOTAL_COUNTS_FILE}")
        print("Normalizzazione non possibile. Usare frequenze raw.")
        return {}
    
    print(f"✓ Caricati total_counts per {len(total_counts)} anni")
    return total_counts


def aggregate_by_decade(input_file: str, total_counts: Dict[int, int]) -> Dict[str, Dict[str, float]]:
    """
    Aggrega n-gram per decennio mantenendo il contesto.
    
    Args:
        input_file: Path del file filtrato
        total_counts: Dizionario total counts per anno
    
    Returns:
        Dizionario {decennio: {ngram: normalized_freq}}
        dove ngram è "word1 word2 word3" per 3-gram
    """
    print(f"\nAggregazione per decennio da {input_file}...")
    
    # Struttura: {decade: {word: total_count}}
    decade_data = {decade: defaultdict(int) for decade in DECADES}
    
    # Struttura: {decade: total_count_decade}
    decade_totals = {decade: 0 for decade in DECADES}
    
    lines_processed = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            lines_processed += 1
            
            if lines_processed % 1000000 == 0:
                print(f"  Processate {lines_processed//1000000}M righe...")
            
            try:
                parts = line.strip().split('\t')
                if len(parts) != 4:
                    continue
                
                ngram, year_str, match_count_str, _ = parts
                year = int(year_str)
                match_count = int(match_count_str)
                
                # Determina decennio
                decade = get_decade_from_year(year)
                
                if decade not in decade_data:
                    continue
                
                # Mantieni gli n-gram completi per preservare il contesto
                # Non spacchettare in parole singole!
                words = ngram.split()
                
                # Filtra n-gram: tutte le parole devono essere valide
                valid = True
                for word in words:
                    word_check = word.lower() if LOWERCASE else word
                    if (len(word_check) < MIN_TOKEN_LENGTH or 
                        len(word_check) > MAX_TOKEN_LENGTH or 
                        not word_check.isalpha()):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Normalizza n-gram (lowercase se richiesto)
                if LOWERCASE:
                    ngram = ' '.join([w.lower() for w in words])
                
                # Aggrega l'n-gram completo (mantiene contesto)
                decade_data[decade][ngram] += match_count
            
            except (ValueError, IndexError):
                continue
    
    print(f"✓ Processate {lines_processed:,} righe totali")
    
    # Calcola totali per decennio (per normalizzazione)
    # Total counts = numero totale di parole pubblicate in quel decennio
    # Serve per normalizzare: "guerra" con 100K occorrenze negli anni '40
    # ha frequenza relativa diversa rispetto agli anni '10 (meno libri pubblicati)
    print("\nCalcolo totali per decennio...")
    for decade in DECADES:
        decade_start = int(decade[:4])
        decade_end = decade_start + 9
        
        for year in range(decade_start, decade_end + 1):
            if year in total_counts:
                decade_totals[decade] += total_counts[year]
    
    # Normalizza frequenze
    # normalized_freq = count / total_words_in_decade
    # Questo rende comparabili le frequenze tra decenni con volumi diversi
    print("Normalizzazione frequenze...")
    normalized_data = {}
    
    for decade in DECADES:
        normalized_data[decade] = {}
        total = decade_totals[decade]
        
        if total == 0:
            print(f"⚠ Total count = 0 per {decade}, uso frequenze raw")
            total = 1
        
        for word, count in decade_data[decade].items():
            # Normalizza: freq = count / total
            # Esempio: "war" con 1M occorrenze su 100B parole totali = freq 1e-5
            normalized_freq = count / total
            normalized_data[decade][word] = normalized_freq
        
        print(f"  {decade}: {len(normalized_data[decade]):,} parole uniche")
    
    return normalized_data


def filter_by_frequency(decade_data: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Filtra parole con occorrenze totali sotto soglia.
    
    Args:
        decade_data: Dati aggregati per decennio
    
    Returns:
        Dati filtrati
    """
    print(f"\nFiltraggio parole con freq totale < {MIN_CORPUS_OCCURRENCES}...")
    
    # Calcola occorrenze totali per ogni parola (somma su tutti i decenni)
    # Nota: per calcolo preciso servirebbe count raw, ma usiamo approssimazione
    word_total_freq = defaultdict(float)
    
    for decade_words in decade_data.values():
        for word, freq in decade_words.items():
            word_total_freq[word] += freq
    
    # Filtra
    filtered_data = {}
    for decade, decade_words in decade_data.items():
        filtered_data[decade] = {
            word: freq
            for word, freq in decade_words.items()
            if word_total_freq[word] >= MIN_CORPUS_OCCURRENCES / 1e9  # Approssimazione
        }
        
        removed = len(decade_words) - len(filtered_data[decade])
        print(f"  {decade}: rimosse {removed} parole rare")
    
    return filtered_data


def save_processed_data(decade_data: Dict[str, Dict[str, float]]):
    """
    Salva dati processati in file separati per decennio.
    
    Args:
        decade_data: Dati aggregati e normalizzati
    """
    print("\nSalvataggio file processati...")
    
    for decade in DECADES:
        output_file = get_processed_file_path(decade)
        
        # Ordina parole per frequenza (decrescente)
        sorted_words = sorted(
            decade_data[decade].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for word, freq in sorted_words:
                f.write(f"{word}\t{freq:.12e}\n")
        
        print(f"  ✓ {decade}: {len(sorted_words):,} parole salvate in {output_file}")


def main():
    """Main function."""
    
    print("="*70)
    print("PREPROCESSING E AGGREGAZIONE PER DECENNIO")
    print("="*70)
    print(f"Periodo: {START_YEAR}-{END_YEAR}")
    print(f"Decenni: {', '.join(DECADES)}")
    print("="*70 + "\n")
    
    # Crea directory
    create_directories()
    
    # File input (usa NGRAM_TYPE da config)
    input_file = os.path.join(DATA_RAW_DIR, f"{NGRAM_TYPE}gram_filtered.tsv")
    
    if not os.path.exists(input_file):
        print(f"❌ File input non trovato: {input_file}")
        print("Eseguire prima download_ngrams.py (FASE 1)")
        return 1
    
    print(f"Elaborazione file {NGRAM_TYPE}-gram: {input_file}")
    
    # Step 1: Carica total_counts
    total_counts = load_total_counts()
    
    # Step 2: Aggrega per decennio
    decade_data = aggregate_by_decade(input_file, total_counts)
    
    # Step 3: Filtra parole rare
    decade_data = filter_by_frequency(decade_data)
    
    # Step 4: Salva
    save_processed_data(decade_data)
    
    print("\n" + "="*70)
    print("✅ FASE 2 COMPLETATA")
    print("="*70)
    print(f"File salvati in: {DATA_PROCESSED_DIR}")
    print("Prossimo step: eseguire build_vocab.py per costruire vocabolario")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
