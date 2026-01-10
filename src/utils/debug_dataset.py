"""
DEBUG DATASET
Verifica visiva che la Sliding Window funzioni.
Stampa Context -> Target decodificati.
"""
import torch
import json
import os
import sys

# Aggiunge la root al path
sys.path.append(os.getcwd())

from src.dataset import CBOWDataset
from src.config import VOCAB_FILE, UNK_TOKEN

def decode_ids(ids, idx2word):
    words = []
    for idx in ids:
        idx = int(idx)
        if idx == 0:
            words.append("[PAD]") # Rendiamo visibile il padding
        else:
            words.append(idx2word.get(idx, UNK_TOKEN))
    return words

def main():
    print("🔬 Inizializzazione Dataset di Debug (1900s)...")
    # Carichiamo il dataset (nota: ci metterà un attimo a preparare i dati)
    dataset = CBOWDataset(decade="1900s")
    
    # Creiamo mappa inversa ID -> Parola per leggere
    with open(VOCAB_FILE, 'r') as f:
        word2idx = json.load(f)
    idx2word = {v: k for k, v in word2idx.items()}
    
    print(f"\n✅ Dataset Caricato. Totale esempi: {len(dataset):,}")
    print("-" * 60)
    print(f"{'CONTESTO (Window=2)':<50} | {'TARGET':<15}")
    print("-" * 60)

    # Stampiamo i primi 20 esempi
    # Nota: Poiché c'è lo shuffle, potresti vedere frasi diverse, 
    # ma cerca pattern con [PAD]
    for i in range(20):
        target_id, context_tensor = dataset[i]
        
        target_word = decode_ids([target_id], idx2word)[0]
        context_words = decode_ids(context_tensor.tolist(), idx2word)
        
        # Formattazione bella
        ctx_str = f"[{', '.join(context_words)}]"
        print(f"{ctx_str:<50} -> {target_word}")

    print("-" * 60)
    print("Se vedi [PAD] spostarsi, la Sliding Window funziona!")

if __name__ == "__main__":
    main()