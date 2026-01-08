"""
CERCA CONTESTI ORIGINALI
Scansiona il file dataset processato (.txt) per trovare i contesti reali di una parola.

USO:
  Per vedere i primi 20:
  python src/check_context.py --decade 2010s --word mouse

  Per vederli TUTTI (illimitato):
  python src/check_context.py --decade 2010s --word mouse --limit 0
"""
import os
import argparse
import sys

# Aggiungi path per importare config
sys.path.append(os.getcwd())
from config import get_processed_file_path

def search_contexts(decade, target_word, limit=0):
    file_path = get_processed_file_path(decade)
    
    if not os.path.exists(file_path):
        print(f"❌ File non trovato: {file_path}")
        return

    print(f"🔎 Cerco contesti per '{target_word}' in {file_path}...")
    if limit > 0:
        print(f"   (Mostro i primi {limit} risultati)")
    else:
        print(f"   (Mostro TUTTI i risultati)")
    print("-" * 60)
    
    count = 0
    target_word = target_word.lower()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split('\t')
                ngram = parts[0]
                
                # Cerca la parola target dentro l'n-gramma
                words = ngram.split()
                if target_word in [w.lower() for w in words]:
                    
                    # Recupera frequenza se esiste
                    freq = parts[1].strip() if len(parts) > 1 else "N/A"
                    
                    # Stampa risultato
                    print(f"[{freq:>6}] {ngram}")
                    
                    count += 1
                    
                    # Se abbiamo un limite ed è stato raggiunto, fermiamoci
                    if limit > 0 and count >= limit:
                        print("-" * 60)
                        print(f"⚠️  Limite di {limit} raggiunto.")
                        break
                        
    except KeyboardInterrupt:
        print("\n🛑 Ricerca interrotta dall'utente.")

    if count == 0:
        print("❌ Nessun contesto trovato.")
    elif limit == 0 or count < limit:
        print("-" * 60)
        print(f"✅ Trovati {count} contesti totali.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decade", type=str, required=True, help="Es: 1900s")
    parser.add_argument("--word", type=str, required=True, help="Parola da cercare")
    # Nuovo argomento limit (default 20, se 0 = infinito)
    parser.add_argument("--limit", type=int, default=20, help="Numero max risultati (0 = tutti)")
    
    args = parser.parse_args()
    
    search_contexts(args.decade, args.word, args.limit)