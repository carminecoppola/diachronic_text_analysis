"""
VERSIONE 1
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


import json
import os
import torch
import random
from torch.utils.data import Dataset
from typing import List, Tuple

from src.config import (
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

"""






"""
VERSIONE 2
FASE 4: Dataset CBOW - Versione "Full Sliding Window"

STRATEGIA:
Per ogni 5-gramma, generiamo TUTTI i possibili target (non solo quello centrale),
usando il PADDING (indice 0) quando la finestra esce dai bordi.

Input:  [A, B, C, D, E]
Output: 
1. Target A | Context [0, 0, B, C]
2. Target B | Context [0, A, C, D]
3. Target C | Context [A, B, D, E]
... e così via.

Vantaggi:
- Moltiplica i dati di training (da 1x a 5x per riga).
- Impara come si comportano le parole anche all'inizio/fine delle frasi.


import json
import os
import torch
import random
from torch.utils.data import Dataset
from typing import List, Tuple

from src.config import (
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
        
        # 2. Carica i dati (Sliding Window)
        self.samples = self._load_data_as_samples()
        
        print(f"✓ Dataset '{decade}' pronto (Full Sliding Window).")
        print(f"  - Numero totale di esempi generati: {len(self.samples):,}")

    def _load_vocab(self):
        if not os.path.exists(VOCAB_FILE):
            raise FileNotFoundError(f"Vocabolario non trovato: {VOCAB_FILE}")
        with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
            self.word2idx = json.load(f)
        # Assumiamo che 0 sia riservato per il padding nel model.py
        # Se UNK non c'è, lo mappiamo a 1 o un altro indice sicuro, ma NON 0 se possibile.
        # Se nel tuo vocab UNK è 0, va bene lo stesso, il modello ignorerà sia UNK che PAD.
        self.unk_idx = self.word2idx.get(UNK_TOKEN, 0)
        self.pad_idx = 0 

    def _load_data_as_samples(self) -> List[Tuple[int, torch.Tensor]]:
        decade_file = get_processed_file_path(self.decade)
        if not os.path.exists(decade_file):
            raise FileNotFoundError(f"Dati non trovati: {decade_file}")

        print(f"  - Caricamento entità da {decade_file}...")
        
        samples_buffer = [] 
        
        with open(decade_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        random.shuffle(lines)
        print("    Righe mescolate. Inizio generazione Sliding Window...")

        for line in lines:
            line = line.strip()
            if not line: continue
            
            parts = line.split('\t')
            ngram_text = parts[0]
            
            # Gestione Frequenza (ripetizione righe)
            repeats = 1
            if len(parts) > 1:
                try:
                    freq = float(parts[1])
                    if freq > 100: repeats = 2
                    if freq > 1000: repeats = 3
                    if freq > 10000: repeats = 5
                    repeats = min(repeats, 10)
                except:
                    pass
            
            words = ngram_text.split()
            # Scarta se troppo corto (es. meno di 2 parole totali)
            if len(words) < 2: 
                continue

            # Converti parole in ID
            ids = [self.word2idx.get(w.lower(), self.unk_idx) for w in words]
            n = len(ids)

            # --- SLIDING WINDOW LOGIC ---
            # Scorriamo su OGNI parola dell'n-gramma considerandola target
            generated_for_this_line = []
            
            for i in range(n):
                target_id = ids[i]
                
                context_ids = []
                
                # Costruiamo Finestra Sinistra
                for w in range(CONTEXT_WINDOW, 0, -1): # es. 2, 1
                    idx_left = i - w
                    if idx_left < 0:
                        context_ids.append(self.pad_idx) # Padding (0)
                    else:
                        context_ids.append(ids[idx_left])
                
                # Costruiamo Finestra Destra
                for w in range(1, CONTEXT_WINDOW + 1): # es. 1, 2
                    idx_right = i + w
                    if idx_right >= n:
                        context_ids.append(self.pad_idx) # Padding (0)
                    else:
                        context_ids.append(ids[idx_right])
                
                # Convertiamo in tensore
                context_tensor = torch.tensor(context_ids, dtype=torch.long)
                generated_for_this_line.append((target_id, context_tensor))
            
            # Aggiungiamo al buffer globale ripetendo se necessario
            for _ in range(repeats):
                samples_buffer.extend(generated_for_this_line)
                
        # Shuffle finale perché abbiamo generato gruppi contigui
        random.shuffle(samples_buffer)
        
        return samples_buffer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target_id, context_tensor = self.samples[idx]
        return torch.tensor(target_id, dtype=torch.long), context_tensor
"""

"""
VERSIONE 3
FASE 4: Dataset CBOW - Versione "Memory Optimized Tensor"

STRATEGIA:
Mantiene la logica "Full Sliding Window" ma elimina l'overhead delle liste Python.
Invece di una lista di tuple [(target, tensor), ...], costruiamo due
grandi Tensor monolitici:
- all_contexts: [N_samples, 2*WINDOW]
- all_targets:  [N_samples]

Vantaggi:
- RAM: Occupa ~600MB invece di 32GB per il dataset 2000s.
- Velocità: Il passaggio dati alla GPU è diretto (memoria contigua).
"""

import json
import os
import torch
import random
from torch.utils.data import Dataset
from typing import Tuple

from src.config import (
    VOCAB_FILE,
    CONTEXT_WINDOW,
    UNK_TOKEN,
    get_processed_file_path,
)

class CBOWDataset(Dataset):
    def __init__(self, decade: str):
        self.decade = decade
        
        # 1. Carica Vocabolario
        self._load_vocab()
        
        # 2. Carica i dati ottimizzati in Tensor
        self.contexts, self.targets = self._load_data_optimized()
        
        print(f"✓ Dataset '{decade}' pronto (Memory Optimized).")
        print(f"  - Esempi totali: {len(self.targets):,}")
        # Calcolo memoria occupata in MB
        mem_usage = (self.contexts.element_size() * self.contexts.nelement() + 
                     self.targets.element_size() * self.targets.nelement()) / (1024 * 1024)
        print(f"  - RAM stimata Tensori: {mem_usage:.2f} MB")

    def _load_vocab(self):
        if not os.path.exists(VOCAB_FILE):
            raise FileNotFoundError(f"Vocabolario non trovato: {VOCAB_FILE}")
        with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
            self.word2idx = json.load(f)
        self.unk_idx = self.word2idx.get(UNK_TOKEN, 0)
        self.pad_idx = 0 

    def _load_data_optimized(self) -> Tuple[torch.Tensor, torch.Tensor]:
        decade_file = get_processed_file_path(self.decade)
        if not os.path.exists(decade_file):
            raise FileNotFoundError(f"Dati non trovati: {decade_file}")

        print(f"  - Caricamento dati grezzi da {decade_file}...")
        
        # Usiamo liste semplici di interi (molto più leggere di Tensor oggetti)
        # Solo alla fine convertiamo in Tensor unico
        temp_contexts = []
        temp_targets = []
        
        with open(decade_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        random.shuffle(lines)
        print("    Righe mescolate. Generazione Tensori...")

        for line in lines:
            line = line.strip()
            if not line: continue
            
            parts = line.split('\t')
            ngram_text = parts[0]
            
            # Gestione Frequenza
            repeats = 1
            if len(parts) > 1:
                try:
                    freq = float(parts[1])
                    if freq > 100: repeats = 2
                    if freq > 1000: repeats = 3
                    if freq > 10000: repeats = 5
                    repeats = min(repeats, 10)
                except:
                    pass
            
            words = ngram_text.split()
            if len(words) < 2: continue

            # Converti parole in ID
            ids = [self.word2idx.get(w.lower(), self.unk_idx) for w in words]
            n = len(ids)

            # --- SLIDING WINDOW LOGIC ---
            # Calcoliamo i contesti per questa riga
            row_contexts = []
            row_targets = []

            for i in range(n):
                target_id = ids[i]
                current_ctx = []
                
                # Finestra Sinistra
                for w in range(CONTEXT_WINDOW, 0, -1):
                    idx_left = i - w
                    if idx_left < 0:
                        current_ctx.append(self.pad_idx)
                    else:
                        current_ctx.append(ids[idx_left])
                
                # Finestra Destra
                for w in range(1, CONTEXT_WINDOW + 1):
                    idx_right = i + w
                    if idx_right >= n:
                        current_ctx.append(self.pad_idx)
                    else:
                        current_ctx.append(ids[idx_right])
                
                row_contexts.append(current_ctx)
                row_targets.append(target_id)
            
            # Aggiungiamo alle liste temporanee N volte (Frequenza)
            for _ in range(repeats):
                temp_contexts.extend(row_contexts)
                temp_targets.extend(row_targets)

        print("    Conversione in Tensor Unici (questo libera la RAM)...")
        # Qui avviene la magia: Convertiamo tutto in 2 Tensori di interi
        # dtype=torch.long serve per gli indici di Embedding
        tensor_contexts = torch.tensor(temp_contexts, dtype=torch.long)
        tensor_targets = torch.tensor(temp_targets, dtype=torch.long)
        
        return tensor_contexts, tensor_targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Accesso istantaneo alla memoria contigua (velocissimo)
        return self.targets[idx], self.contexts[idx]