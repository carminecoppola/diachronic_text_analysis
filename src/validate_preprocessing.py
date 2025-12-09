"""
Validazione Risultati Preprocessing

Questo script valida i risultati del preprocessing:
- Controlla che tutti i file siano stati generati
- Analizza statistiche descrittive
- Identifica eventuali anomalie
"""

from pathlib import Path
from collections import Counter
import config


def load_tokens(period):
    """Carica i token processati per un periodo."""
    file_path = config.get_period_file_path(period, 'processed')
    
    if not file_path.exists():
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        tokens = [line.strip() for line in f if line.strip()]
    
    return tokens


def validate_preprocessing():
    """Valida il preprocessing di tutti i periodi."""
    print("="*70)
    print("  VALIDAZIONE PREPROCESSING")
    print("="*70)
    print()
    
    all_stats = []
    all_vocab = set()
    
    for period in config.PERIODS:
        print(f"[{period}]")
        
        # Carica token
        tokens = load_tokens(period)
        
        if tokens is None:
            print(f"  ✗ File non trovato!")
            continue
        
        # Statistiche
        vocab = set(tokens)
        counter = Counter(tokens)
        
        stats = {
            'period': period,
            'total_tokens': len(tokens),
            'unique_tokens': len(vocab),
            'top_10': counter.most_common(10),
            'shortest': min(tokens, key=len),
            'longest': max(tokens, key=len),
            'avg_length': sum(len(t) for t in tokens) / len(tokens)
        }
        
        all_vocab.update(vocab)
        all_stats.append(stats)
        
        # Output
        print(f"  Tokens totali: {stats['total_tokens']:,}")
        print(f"  Vocabolario: {stats['unique_tokens']} parole uniche")
        print(f"  Lunghezza media token: {stats['avg_length']:.2f} caratteri")
        print(f"  Token più corto: '{stats['shortest']}' ({len(stats['shortest'])} char)")
        print(f"  Token più lungo: '{stats['longest']}' ({len(stats['longest'])} char)")
        print(f"  Top 5 parole: {[w for w, c in stats['top_10'][:5]]}")
        print()
    
    # Statistiche globali
    print("="*70)
    print("  STATISTICHE GLOBALI")
    print("="*70)
    print()
    
    total_tokens = sum(s['total_tokens'] for s in all_stats)
    print(f"Tokens totali (tutti i periodi): {total_tokens:,}")
    print(f"Vocabolario totale: {len(all_vocab)} parole uniche")
    print(f"Media tokens per periodo: {total_tokens / len(config.PERIODS):,.0f}")
    print()
    
    # Analisi crescita vocabolario nel tempo
    print("Crescita vocabolario nel tempo:")
    print(f"{'Periodo':<10} {'Vocab Size':<12} {'Variazione':<12}")
    print("-" * 50)
    
    prev_vocab_size = None
    for stats in all_stats:
        vocab_size = stats['unique_tokens']
        
        if prev_vocab_size is not None:
            delta = vocab_size - prev_vocab_size
            delta_str = f"+{delta}" if delta >= 0 else str(delta)
        else:
            delta_str = "—"
        
        print(f"{stats['period']:<10} {vocab_size:<12} {delta_str:<12}")
        prev_vocab_size = vocab_size
    
    print()
    
    # Controllo qualità
    print("="*70)
    print("  CONTROLLO QUALITÀ")
    print("="*70)
    print()
    
    issues = []
    
    # Verifica token troppo corti o lunghi
    for stats in all_stats:
        if len(stats['shortest']) < config.MIN_TOKEN_LENGTH:
            issues.append(f"{stats['period']}: Token troppo corto: '{stats['shortest']}'")
        
        if len(stats['longest']) > config.MAX_TOKEN_LENGTH:
            issues.append(f"{stats['period']}: Token troppo lungo: '{stats['longest']}'")
    
    # Verifica distribuzione uniforme
    avg_tokens = total_tokens / len(config.PERIODS)
    for stats in all_stats:
        variance = abs(stats['total_tokens'] - avg_tokens) / avg_tokens
        if variance > 0.1:  # >10% differenza
            issues.append(f"{stats['period']}: Deviazione significativa nel numero di tokens ({variance:.1%})")
    
    if issues:
        print("⚠️  Problemi rilevati:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Nessun problema rilevato!")
    
    print()
    
    # Esempi di parole per verifica manuale
    print("="*70)
    print("  CAMPIONE VOCABOLARIO (per verifica manuale)")
    print("="*70)
    print()
    
    sample_words = sorted(all_vocab)[:50]
    print("Prime 50 parole (alfabeticamente):")
    for i in range(0, len(sample_words), 10):
        print("  ", ", ".join(sample_words[i:i+10]))
    
    print()
    print("✅ Validazione completata!")
    print()


if __name__ == "__main__":
    validate_preprocessing()
