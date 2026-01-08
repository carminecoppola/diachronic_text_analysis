"""
FASE 4: Dataset CBOW - Versione "Entità Separate"

STRATEGIA:
Invece di un unico tensore piatto, creiamo una lista di esempi pre-calcolati.
Ogni elemento della lista è una tupla (target, contesto).

Vantaggi:
- Impossibile mischiare frasi diverse (sono oggetti separati in memoria).
- Shuffle perfetto.
- Logica cristallina.

Requisiti:
- Adatto per dataset fino a ~5-10 Milioni di trigrammi (per la RAM).
- Con 750k trigrammi occupa pochissimo (qualche centinaio di MB).
"""

import json
import os
import torch
import random
from torch.utils.data import Dataset
from typing import List, Tuple

from config import (
    VOCAB_FILE,
    CONTEXT_WINDOW,
    UNK_TOKEN,
    get_processed_file_path,
)

class CBOWDataset(Dataset):
    def __init__(self, decade: str, cache_contexts: bool = False):
        self.decade = decade
        
        # 1. Carica Vocabolario
        self._load_vocab()
        
        # 2. Carica i dati come entità separate
        # self.samples sarà una lista di tuple: [(target_id, context_tensor), ...]
        self.samples = self._load_data_as_samples()
        
        print(f"✓ Dataset '{decade}' pronto (Modalità Entità Separate).")
        print(f"  - Numero totale di esempi pronti per il training: {len(self.samples):,}")

    def _load_vocab(self):
        if not os.path.exists(VOCAB_FILE):
            raise FileNotFoundError(f"Vocabolario non trovato: {VOCAB_FILE}")
        with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
            self.word2idx = json.load(f)
        self.unk_idx = self.word2idx.get(UNK_TOKEN, 0)

    def _load_data_as_samples(self) -> List[Tuple[int, torch.Tensor]]:
        decade_file = get_processed_file_path(self.decade)
        if not os.path.exists(decade_file):
            raise FileNotFoundError(f"Dati non trovati: {decade_file}")

        print(f"  - Caricamento entità da {decade_file}...")
        
        samples_buffer = [] 
        
        with open(decade_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # SHUFFLE INIZIALE DELLE RIGHE GREZZE
        # Mischiamo l'ordine di lettura dal file
        random.shuffle(lines)
        print("    Righe mescolate. Inizio elaborazione...")

        for line in lines:
            line = line.strip()
            if not line: continue
            
            parts = line.split('\t')
            ngram_text = parts[0]
            
            # --- GESTIONE FREQUENZA ---
            # Più è frequente, più volte aggiungiamo lo stesso esempio alla lista
            repeats = 1
            if len(parts) > 1:
                try:
                    freq = float(parts[1])
                    if freq > 100: repeats = 2
                    if freq > 1000: repeats = 3
                    if freq > 10000: repeats = 5
                    repeats = min(repeats, 10) # Tetto massimo
                except:
                    pass
            
            words = ngram_text.split()
            # Se abbiamo meno parole del necessario, scartiamo
            # Es: window=1 serve 1(sx) + 1(centro) + 1(dx) = 3 parole
            if len(words) < 2 * CONTEXT_WINDOW + 1:
                continue

            # Converti parole in ID
            ids = [self.word2idx.get(w.lower(), self.unk_idx) for w in words]
            
            # Estraiamo SOLO la parola centrale come target
            # Assumiamo che nei trigrammi ci sia un solo centro valido
            center_pos = len(ids) // 2 
            
            # Verifica che il centro abbia abbastanza contesto a sx e dx
            if center_pos < CONTEXT_WINDOW or (center_pos + CONTEXT_WINDOW) >= len(ids):
                continue
                
            target_id = ids[center_pos]
            
            # Costruiamo il contesto (indipendente!)
            # Prende le parole a sx e a dx del centro
            start = center_pos - CONTEXT_WINDOW
            end = center_pos + CONTEXT_WINDOW + 1
            
            # Slice pura di Python (crea una nuova lista, nessuna referenza strana)
            context_list = ids[start:center_pos] + ids[center_pos+1:end]
            context_tensor = torch.tensor(context_list, dtype=torch.long)
            
            # Aggiungiamo alla lista "repeats" volte
            for _ in range(repeats):
                samples_buffer.append((target_id, context_tensor))
                
        # SHUFFLE FINALE DEGLI ESEMPI
        # Mischiamo di nuovo perché abbiamo duplicato le righe frequenti vicine
        random.shuffle(samples_buffer)
        
        return samples_buffer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Qui è semplicissimo: prendiamo l'oggetto già pronto dalla lista
        target_id, context_tensor = self.samples[idx]
        
        # Ritorniamo target come tensore scalare
        return torch.tensor(target_id, dtype=torch.long), context_tensor