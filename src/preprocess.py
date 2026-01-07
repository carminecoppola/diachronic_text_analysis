"""
FASE 2: Preprocessing e aggregazione per decennio

Questo script processa i dati filtrati da FASE 1 e li aggrega per decennio.

COSA FA:
1. Legge data/raw/Ngram_filtered.tsv (supporta 1-gram e 5-gram)
2. Aggrega frequenze per decennio (1900-1909 → 1900s, ecc.)
3. Normalizza frequenze usando total_counts
4. Applica filtri di pulizia (lunghezza, lowercase)
5. Salva file separati per ogni decennio in data/processed/

INPUT:
- data/raw/1gram_filtered.tsv o data/raw/5gram_filtered.tsv
- data/raw/total_counts.txt

OUTPUT:
- data/processed/1900s.txt
- data/processed/1910s.txt
- ...
- data/processed/1990s.txt

Formato output: ogni file contiene righe con:
word \t normalized_frequency

NOTA PER 5-GRAM:
Per 5-gram, ogni riga contiene "word1 word2 word3 word4 word5".
Questo script estrae e aggrega TUTTE le 5 parole separatamente,
in modo che il vocabolario contenga parole singole.
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
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    year = int(parts[0])
                    count = int(parts[1])
                    total_counts[year] = count
    
    except FileNotFoundError:
        print(f"⚠ File total_counts non trovato: {TOTAL_COUNTS_FILE}")
        print("Normalizzazione non possibile. Usare frequenze raw.")
        return {}
    
    print(f"✓ Caricati total_counts per {len(total_counts)} anni")
    return total_counts


def aggregate_by_decade(input_file: str, total_counts: Dict[int, int]) -> Dict[str, Dict[str, float]]:
    """
    Aggrega n-gram per decennio.
    
    Args:
        input_file: Path del file filtrato
        total_counts: Dizionario total counts per anno
    
    Returns:
        Dizionario {decennio: {word: normalized_freq}}
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
                
                # Per 5-gram: estrai tutte le 5 parole
                # Per 1-gram: una sola parola
                words = ngram.split() if NGRAM_TYPE > 1 else [ngram]
                
                # Processa ogni parola nell'n-gram
                for word in words:
                    # Applica filtri preprocessing
                    # 1. Lowercase
                    if LOWERCASE:
                        word = word.lower()
                    
                    # 2. Lunghezza
                    if len(word) < MIN_TOKEN_LENGTH or len(word) > MAX_TOKEN_LENGTH:
                        continue
                    
                    # 3. Solo alfabetici
                    if not word.isalpha():
                        continue
                    
                    # Aggrega (dividiamo il count equamente tra le parole per 5-gram)
                    # Ogni parola nel 5-gram contribuisce con count/5 alla sua frequenza
                    count_per_word = match_count // max(len(words), 1)
                    decade_data[decade][word] += count_per_word
            
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
