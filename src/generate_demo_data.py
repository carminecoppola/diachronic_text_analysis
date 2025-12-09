"""
Generazione Dataset Demo per Testing

Questo script genera un dataset sintetico demo per testare la pipeline
senza dover scaricare GB di dati da Google Books N-grams.

Il dataset include parole che simulano semantic drift:
- Parole stabili (es. 'water', 'tree')
- Parole con drift tecnologico (es. 'cell', 'mouse', 'web')
- Parole con drift sociale (es. 'gay', 'woke')
"""

import random
from pathlib import Path
from collections import defaultdict
import config


# Seed per riproducibilità
random.seed(42)

# Vocabolario base (parole stabili)
STABLE_WORDS = [
    'water', 'tree', 'house', 'book', 'sun', 'moon', 'day', 'night',
    'food', 'animal', 'person', 'child', 'woman', 'man', 'family',
    'city', 'country', 'world', 'time', 'year', 'life', 'death',
    'love', 'peace', 'war', 'king', 'queen', 'god', 'church',
    'walk', 'run', 'eat', 'drink', 'sleep', 'think', 'know', 'say',
    'good', 'bad', 'big', 'small', 'new', 'old', 'long', 'short',
]

# Parole con drift tecnologico (aumentano nel tempo)
TECH_DRIFT_WORDS = {
    '1900s': ['telephone', 'telegraph', 'railroad', 'steam'],
    '1910s': ['automobile', 'airplane', 'radio', 'electric'],
    '1920s': ['broadcast', 'cinema', 'recording', 'wireless'],
    '1930s': ['television', 'aviation', 'electronics', 'transmission'],
    '1940s': ['radar', 'computer', 'atomic', 'jet'],
    '1950s': ['transistor', 'nuclear', 'satellite', 'automation'],
    '1960s': ['space', 'satellite', 'mainframe', 'programming'],
    '1970s': ['microprocessor', 'calculator', 'software', 'database'],
    '1980s': ['personal', 'computer', 'disk', 'modem', 'digital'],
    '1990s': ['internet', 'web', 'email', 'online', 'browser', 'mouse'],
}

# Parole con semantic shift (cambio di contesto)
SEMANTIC_SHIFT_WORDS = {
    # 'gay': felice -> orientamento sessuale
    'gay': {
        '1900s': ['happy', 'cheerful', 'merry', 'bright', 'joyful'] * 100,
        '1910s': ['happy', 'cheerful', 'merry', 'festive'] * 90,
        '1920s': ['happy', 'cheerful', 'bright'] * 80,
        '1930s': ['happy', 'cheerful', 'homosexual'] * 70,
        '1940s': ['happy', 'homosexual', 'different'] * 60,
        '1950s': ['homosexual', 'different', 'happy'] * 50,
        '1960s': ['homosexual', 'pride', 'community'] * 60,
        '1970s': ['homosexual', 'rights', 'pride', 'community'] * 70,
        '1980s': ['homosexual', 'rights', 'pride', 'lgbtq'] * 80,
        '1990s': ['homosexual', 'rights', 'pride', 'lgbtq', 'community'] * 90,
    },
    # 'cell': cellula -> telefono cellulare
    'cell': {
        '1900s': ['biology', 'organism', 'tissue', 'prison'] * 100,
        '1910s': ['biology', 'organism', 'tissue', 'prison'] * 100,
        '1920s': ['biology', 'organism', 'tissue', 'prison'] * 100,
        '1930s': ['biology', 'organism', 'tissue', 'prison'] * 100,
        '1940s': ['biology', 'organism', 'tissue', 'prison'] * 100,
        '1950s': ['biology', 'organism', 'tissue', 'prison'] * 100,
        '1960s': ['biology', 'organism', 'tissue', 'prison'] * 100,
        '1970s': ['biology', 'organism', 'tissue', 'prison'] * 90,
        '1980s': ['biology', 'phone', 'mobile', 'wireless', 'prison'] * 50,
        '1990s': ['phone', 'mobile', 'wireless', 'cellular', 'biology'] * 100,
    },
    # 'mouse': topo -> dispositivo computer
    'mouse': {
        '1900s': ['animal', 'rodent', 'small', 'pest'] * 100,
        '1910s': ['animal', 'rodent', 'small', 'pest'] * 100,
        '1920s': ['animal', 'rodent', 'small', 'pest'] * 100,
        '1930s': ['animal', 'rodent', 'small', 'pest'] * 100,
        '1940s': ['animal', 'rodent', 'small', 'pest'] * 100,
        '1950s': ['animal', 'rodent', 'small', 'pest'] * 100,
        '1960s': ['animal', 'rodent', 'small', 'computer'] * 90,
        '1970s': ['animal', 'rodent', 'computer', 'pointer'] * 70,
        '1980s': ['computer', 'pointer', 'click', 'device', 'animal'] * 60,
        '1990s': ['computer', 'click', 'pointer', 'device', 'interface'] * 100,
    },
}


def generate_period_corpus(period, num_sentences=5000):
    """
    Genera un corpus per un periodo specifico.
    
    Args:
        period: Nome del periodo (es. '1900s')
        num_sentences: Numero di frasi da generare
    
    Returns:
        Lista di parole per il periodo
    """
    corpus = []
    
    # Aggiungi parole stabili (frequenza costante)
    for _ in range(num_sentences * 10):
        corpus.extend(random.choices(STABLE_WORDS, k=random.randint(5, 15)))
    
    # Aggiungi parole tecnologiche (aumentano nel tempo)
    if period in TECH_DRIFT_WORDS:
        tech_words = TECH_DRIFT_WORDS[period]
        period_idx = list(TECH_DRIFT_WORDS.keys()).index(period)
        frequency_multiplier = (period_idx + 1) * 100  # Aumenta nel tempo
        
        for _ in range(frequency_multiplier):
            corpus.extend(random.choices(tech_words, k=random.randint(1, 3)))
    
    # Aggiungi parole con semantic shift
    for word, contexts in SEMANTIC_SHIFT_WORDS.items():
        if period in contexts:
            corpus.append(word)
            corpus.extend(contexts[period])
    
    return corpus


def save_demo_corpus():
    """
    Genera e salva corpus demo per tutti i periodi.
    """
    print("="*70)
    print("  Generazione Dataset Demo per Testing")
    print("="*70)
    print()
    
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    for period in config.PERIODS:
        print(f"Generazione corpus per {period}...")
        
        # Genera corpus
        corpus = generate_period_corpus(period)
        
        # Salva in file
        output_file = config.RAW_DATA_DIR / f"{period}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(' '.join(corpus))
        
        stats[period] = {
            'total_tokens': len(corpus),
            'unique_words': len(set(corpus))
        }
        
        print(f"  ✓ {period}: {stats[period]['total_tokens']} tokens, "
              f"{stats[period]['unique_words']} parole uniche")
    
    print()
    print("="*70)
    print("  ✅ Dataset Demo Generato!")
    print("="*70)
    print(f"File salvati in: {config.RAW_DATA_DIR}")
    print()
    print("Parole target con semantic drift incluse:")
    print("  - 'gay': happy -> homosexual")
    print("  - 'cell': biology/prison -> mobile phone")
    print("  - 'mouse': animal -> computer device")
    print()
    print("Usa questi dati per testare l'intera pipeline.")
    print()


if __name__ == "__main__":
    save_demo_corpus()
