#!/usr/bin/env python3
"""
Script di Verifica Configurazione Progetto

Controlla che tutto sia pronto per procedere con le fasi successive.
"""

import sys
from pathlib import Path

def check_structure():
    """Verifica struttura cartelle."""
    print("="*70)
    print("  VERIFICA STRUTTURA PROGETTO")
    print("="*70)
    print()
    
    required_dirs = [
        'data/raw',
        'data/processed',
        'models',
        'alignment',
        'plots',
        'notebooks',
        'src'
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - MANCANTE!")
            all_ok = False
    
    return all_ok


def check_dependencies():
    """Verifica dipendenze installate."""
    print()
    print("="*70)
    print("  VERIFICA DIPENDENZE")
    print("="*70)
    print()
    
    deps = [
        'torch',
        'numpy',
        'pandas',
        'nltk',
        'scipy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'tqdm'
    ]
    
    all_ok = True
    for dep in deps:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - NON INSTALLATO!")
            all_ok = False
    
    return all_ok


def check_data():
    """Verifica dati preprocessati."""
    print()
    print("="*70)
    print("  VERIFICA DATI PREPROCESSATI")
    print("="*70)
    print()
    
    processed_dir = Path('data/processed')
    
    if not processed_dir.exists():
        print("❌ Cartella data/processed/ non esiste!")
        return False
    
    files = list(processed_dir.glob('*.txt'))
    
    if not files:
        print("⚠️  Nessun file processato trovato!")
        print("   Esegui: python src/preprocess.py")
        return False
    
    print(f"✅ Trovati {len(files)} file processati:")
    for f in sorted(files):
        size = f.stat().st_size
        lines = sum(1 for _ in open(f))
        print(f"   - {f.name}: {lines:,} tokens ({size/1024:.1f} KB)")
    
    return True


def check_config():
    """Verifica configurazione."""
    print()
    print("="*70)
    print("  VERIFICA CONFIGURAZIONE")
    print("="*70)
    print()
    
    sys.path.insert(0, 'src')
    
    try:
        import config
        
        print(f"✅ NLP Tool: {config.NLP_TOOL}")
        print(f"✅ Periodi: {len(config.PERIODS)}")
        print(f"✅ Embedding Dim: {config.EMBEDDING_DIM}")
        print(f"✅ Model Type: {config.MODEL_TYPE}")
        print(f"✅ Batch Size: {config.BATCH_SIZE}")
        
        return True
    except Exception as e:
        print(f"❌ Errore importazione config: {e}")
        return False


def main():
    """Main check."""
    print()
    print("🔍 VERIFICA CONFIGURAZIONE PROGETTO DIACHRONIC TEXT ANALYSIS")
    print()
    
    checks = {
        'Struttura': check_structure(),
        'Dipendenze': check_dependencies(),
        'Configurazione': check_config(),
        'Dati': check_data()
    }
    
    print()
    print("="*70)
    print("  RIEPILOGO")
    print("="*70)
    print()
    
    for check_name, result in checks.items():
        status = "✅ OK" if result else "❌ ERRORE"
        print(f"{check_name:<20} {status}")
    
    print()
    
    if all(checks.values()):
        print("🎉 TUTTO PRONTO! Puoi procedere con la Fase 3.")
        print()
        print("Prossimi passi:")
        print("  1. Apri il notebook: jupyter notebook notebooks/01_intro_e_preprocessing.ipynb")
        print("  2. Oppure procedi con: python src/build_vocab.py")
        return 0
    else:
        print("⚠️  Alcuni controlli sono falliti. Sistema gli errori prima di procedere.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
