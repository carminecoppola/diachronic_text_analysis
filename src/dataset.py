"""
FASE 4: PyTorch Dataset per CBOW Training

Questo modulo implementa un PyTorch Dataset che:
1. Carica i dati processati per un decennio (LAZY - streaming)
2. Genera contesti CBOW (parola target + contesto circostante) on-the-fly
3. Fornisce batch iterator per il training

ARCHITETTURA:
Due implementazioni fornite:

1. CBOWDataset (CONSIGLIATO per large datasets):
   - Modalità LAZY: carica token dal file, genera contesti on-demand
   - Non cachea i contesti in memoria
   - Ideale per dataset da 100M+ token
   - Memoria: O(context_window) - molto bassa

2. CBOWDatasetCached (per dataset piccoli):
   - Cachea tutti i contesti in memoria prima
   - Più veloce per dataset piccoli (<10M token)
   - Memoria: O(num_contexts) - molto alta

FORMATI:
- Input: data/processed/XXXX.txt (parola<tab>frequenza per riga)
- Output per __getitem__:
  * targets: tensor [batch_size] indici parole target
  * contexts: tensor [batch_size, 2*CONTEXT_WINDOW] indici contesto

ESEMPIO (LAZY - consigliato):
```python
from dataset import CBOWDataset
from config import CONTEXT_WINDOW
from torch.utils.data import DataLoader

# Crea dataset in modalità LAZY (non cachea)
dataset = CBOWDataset(decade='1950s', cache_contexts=False)
print(f"Contesti stimati: {dataset.estimated_size:,}")

# DataLoader per batching
loader = DataLoader(dataset, batch_size=128, shuffle=True)
for targets, contexts in loader:
    print(f"Batch target shape: {targets.shape}")
    print(f"Batch context shape: {contexts.shape}")
    break
```

PARAMETRI CONFIGURABILI (in src/config.py):
- CONTEXT_WINDOW: dimensione finestra di contesto (5 = ±5 parole)
- VOCAB_FILE: path del vocabolario
- DATA_PROCESSED_DIR: directory con i file processati
"""

import json
import os
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
import array as arr

from config import (
    DATA_PROCESSED_DIR,
    VOCAB_FILE,
    CONTEXT_WINDOW,
    UNK_TOKEN,
    DECADES,
    get_processed_file_path,
)


class CBOWDataset(Dataset):
    """
    PyTorch Dataset per training CBOW (Continuous Bag of Words) - MODALITÀ LAZY.
    
    Questo dataset NON cachea i contesti in memoria, ma li genera on-demand
    mentre vengono richiesti. Ideale per dataset molto grandi (100M+ token).
    
    Attributes:
        decade (str): Decennio da cui caricare i dati (es. "1950s")
        word2idx (Dict[str, int]): Mapping parola → indice vocabolario
        tokens (List[int]): Lista di token numerici del decennio
        estimated_size (int): Stima del numero di contesti disponibili
    """
    
    def __init__(self, decade: str, cache_contexts: bool = False):
        """
        Inizializza il dataset per un decennio specifico.
        
        Args:
            decade (str): Decennio da caricare (es. "1950s")
            cache_contexts (bool): Se True, usa CBOWDatasetCached (deprecato)
        
        Raises:
            FileNotFoundError: Se il file della decade o vocabolario non trovato
            ValueError: Se decade non è in format valido
        """
        if cache_contexts:
            # Fallback a versione cached (deprecato)
            print("⚠️  cache_contexts=True è deprecato. Usa CBOWDatasetCached direttamente.")
        
        self.decade = decade
        
        # Validazione decennio
        if decade not in DECADES:
            raise ValueError(
                f"Decennio '{decade}' non valido. "
                f"Validi: {DECADES}"
            )
        
        # Carica vocabolario
        self.word2idx = self._load_vocab()
        unk_idx = self.word2idx.get(UNK_TOKEN, 0)
        
        # Carica dati del decennio (LAZY - solo tokens, non contesti)
        self.tokens = self._load_decade_tokens(unk_idx)
        
        # Stima numero di contesti
        self.estimated_size = max(0, len(self.tokens) - 2 * CONTEXT_WINDOW)
        
        print(f"✓ Dataset '{decade}' caricato (LAZY mode):")
        print(f"  - Tokens totali: {len(self.tokens):,}")
        print(f"  - Contesti stimati: {self.estimated_size:,}")
    
    def _load_vocab(self) -> Dict[str, int]:
        """
        Carica il vocabolario da file JSON.
        
        Returns:
            Dict mapping parola → indice
        
        Raises:
            FileNotFoundError: Se VOCAB_FILE non esiste
        """
        if not os.path.exists(VOCAB_FILE):
            raise FileNotFoundError(
                f"File vocabolario non trovato: {VOCAB_FILE}\n"
                f"Esegui prima build_vocab.py"
            )
        
        with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
            word2idx = json.load(f)
        
        print(f"✓ Vocabolario caricato: {len(word2idx):,} parole")
        return word2idx
    
    def _load_decade_tokens(self, unk_idx: int) -> List[int]:
        """
        Carica i token del decennio dal file processato.
        
        Formato del file:
        parola<tab>frequenza
        parola<tab>frequenza
        ...
        
        La frequenza indica il numero di volte che la parola
        appare nel corpus, quindi replicheremo ogni parola
        freq volte nel token stream.
        
        Args:
            unk_idx (int): Indice da usare per parole non nel vocabolario
        
        Returns:
            List di token numerici
        
        Raises:
            FileNotFoundError: Se il file del decennio non esiste
        """
        decade_file = get_processed_file_path(self.decade)
        
        if not os.path.exists(decade_file):
            raise FileNotFoundError(
                f"File decennio non trovato: {decade_file}\n"
                f"Esegui prima download_ngrams.py e preprocess.py"
            )
        
        # tokens = []
        tokens = arr.array('I')  # Usa array di interi per efficienza memoria
        with open(decade_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) != 2:
                    continue
                
                word, freq_str = parts
                try:
                    freq = int(float(freq_str))
                except ValueError:
                    continue
                
                # Ottieni indice parola (UNK se non nel vocabolario)
                word_idx = self.word2idx.get(word.lower(), unk_idx)
                
                # Replica la parola secondo la sua frequenza
                # Cappare freq massima a 10000 per evitare OOM
                freq = min(freq, 10000)
                tokens.extend([word_idx] * freq)
                
                # Progress ogni 10k righe
                if (line_idx + 1) % 10000 == 0:
                    print(f"  - Processate {line_idx + 1:,} righe...")
        
        return tokens
    
    def __len__(self) -> int:
        """
        Restituisce il numero stimato di contesti nel dataset.
        
        Nota: In modalità LAZY, questo è una stima. Il numero effettivo
        potrebbe variare leggermente a causa di edge cases.
        
        Returns:
            Numero stimato di contesti
        """
        return self.estimated_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Restituisce un singolo contesto CBOW.
        
        Genera il contesto on-demand, non cachea.
        
        Args:
            idx (int): Indice del contesto (0 <= idx < len(dataset))
        
        Returns:
            Tuple di due tensori:
            - target (int64): indice della parola target (scalare)
            - context (int64): indici del contesto [2*CONTEXT_WINDOW]
        
        Raises:
            IndexError: Se idx è fuori range
        """
        if idx >= self.estimated_size:
            raise IndexError(
                f"Indice {idx} fuori range [0, {self.estimated_size - 1}]"
            )
        
        # Posizione del target nel token stream
        target_pos = CONTEXT_WINDOW + idx
        
        # Estrai target e contesto
        target_idx = self.tokens[target_pos]
        
        context_start = target_pos - CONTEXT_WINDOW
        context_end = target_pos + CONTEXT_WINDOW + 1
        context_indices = (
            self.tokens[context_start:target_pos] +  # Sinistra
            self.tokens[target_pos + 1:context_end]   # Destra
        )
        
        # Validazione indices: assicura che siano in range [0, vocab_size)
        vocab_size = len(self.word2idx)
        target_idx = max(0, min(target_idx, vocab_size - 1))
        context_indices = [max(0, min(c, vocab_size - 1)) for c in context_indices]
        
        # Converti in tensori PyTorch
        target = torch.tensor(target_idx, dtype=torch.int64)
        context = torch.tensor(context_indices, dtype=torch.int64)
        
        return target, context


class CBOWDatasetCached(Dataset):
    """
    PyTorch Dataset per training CBOW - MODALITÀ CACHED (deprecato).
    
    Cachea TUTTI i contesti in memoria PRIMA. Usa SOLO per dataset piccoli (<10M token).
    
    WARNING: Questo può causare OOM su dataset grandi!
    """
    
    def __init__(self, decade: str):
        """
        Inizializza il dataset in modalità cached.
        
        ⚠️  WARNING: Cachea tutti i contesti in memoria!
        """
        self.decade = decade
        
        # Validazione decennio
        if decade not in DECADES:
            raise ValueError(
                f"Decennio '{decade}' non valido. "
                f"Validi: {DECADES}"
            )
        
        # Carica vocabolario
        self.word2idx = self._load_vocab()
        unk_idx = self.word2idx.get(UNK_TOKEN, 0)
        
        # Carica dati del decennio
        self.tokens = self._load_decade_tokens(unk_idx)
        
        # Genera tutti i contesti (ATTENZIONE: può usare molte RAM!)
        print(f"⚠️  Generando e cachando contesti in memoria (potrebbe essere lento)...")
        self.contexts_list = self._generate_contexts()
        
        print(f"✓ Dataset '{decade}' caricato (CACHED mode):")
        print(f"  - Tokens totali: {len(self.tokens):,}")
        print(f"  - Contesti cachati: {len(self.contexts_list):,}")
    
    def _load_vocab(self) -> Dict[str, int]:
        """Carica il vocabolario da file JSON."""
        if not os.path.exists(VOCAB_FILE):
            raise FileNotFoundError(
                f"File vocabolario non trovato: {VOCAB_FILE}\n"
                f"Esegui prima build_vocab.py"
            )
        
        with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
            word2idx = json.load(f)
        
        print(f"✓ Vocabolario caricato: {len(word2idx):,} parole")
        return word2idx
    
    def _load_decade_tokens(self, unk_idx: int) -> List[int]:
        """Carica i token del decennio dal file processato."""
        decade_file = get_processed_file_path(self.decade)
        
        if not os.path.exists(decade_file):
            raise FileNotFoundError(
                f"File decennio non trovato: {decade_file}\n"
                f"Esegui prima download_ngrams.py e preprocess.py"
            )
        
        tokens = []
        with open(decade_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) != 2:
                    continue
                
                word, freq_str = parts
                try:
                    freq = int(float(freq_str))
                except ValueError:
                    continue
                
                word_idx = self.word2idx.get(word.lower(), unk_idx)
                freq = min(freq, 10000)
                tokens.extend([word_idx] * freq)
                
                if (line_idx + 1) % 10000 == 0:
                    print(f"  - Processate {line_idx + 1:,} righe...")
        
        return tokens
    
    def _generate_contexts(self) -> List[Tuple[int, List[int]]]:
        """
        Genera TUTTI i contesti CBOW e li cachea in memoria.
        
        ⚠️  WARNING: Questo può causare OOM su dataset grandi!
        """
        contexts = []
        
        for target_pos in range(CONTEXT_WINDOW, len(self.tokens) - CONTEXT_WINDOW):
            target_idx = self.tokens[target_pos]
            
            context_start = target_pos - CONTEXT_WINDOW
            context_end = target_pos + CONTEXT_WINDOW + 1
            context_indices = (
                self.tokens[context_start:target_pos] +
                self.tokens[target_pos + 1:context_end]
            )
            
            contexts.append((target_idx, context_indices))
            
            if (len(contexts) + 1) % 100000 == 0:
                print(f"  - Generati {len(contexts):,} contesti...")
        
        return contexts
    
    def __len__(self) -> int:
        """Restituisce il numero totale di contesti nel dataset."""
        return len(self.contexts_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Restituisce un singolo contesto CBOW."""
        if idx >= len(self.contexts_list):
            raise IndexError(
                f"Indice {idx} fuori range [0, {len(self.contexts_list) - 1}]"
            )
        
        target_idx, context_indices = self.contexts_list[idx]
        
        target = torch.tensor(target_idx, dtype=torch.int64)
        context = torch.tensor(context_indices, dtype=torch.int64)
        
        return target, context
# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_vocab_size() -> int:
    """
    Ottiene la dimensione del vocabolario.
    
    Returns:
        Numero di token nel vocabolario
    """
    with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return len(vocab)


def get_vocab_dict() -> Dict[str, int]:
    """
    Carica il vocabolario completo.
    
    Returns:
        Dict mapping parola → indice
    """
    with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab


def get_reverse_vocab_dict() -> Dict[int, str]:
    """
    Carica il vocabolario inverso (indice → parola).
    
    Returns:
        Dict mapping indice → parola
    """
    vocab = get_vocab_dict()
    return {idx: word for word, idx in vocab.items()}


# ============================================================================
# TEST & DEBUG
# ============================================================================

if __name__ == "__main__":
    """
    Test del dataset con un decennio di esempio (versione leggera).
    """
    print("🧪 Test Dataset CBOW (LAZY mode)\n")
    
    try:
        # Testa caricamento vocabolario
        print("📂 Test 1: Caricamento vocabolario...")
        vocab = get_vocab_dict()
        print(f"✓ Vocabolario: {len(vocab):,} token")
        
        reverse_vocab = get_reverse_vocab_dict()
        print(f"✓ Vocabolario inverso: {len(reverse_vocab):,} token")
        
        # Mostra primi token
        sample_words = list(vocab.keys())[:10]
        print(f"  - Sample parole: {sample_words}")
        
        # Test parametri dataset
        print(f"\n📊 Test 2: Parametri dataset")
        print(f"  - CONTEXT_WINDOW: {CONTEXT_WINDOW}")
        print(f"  - Dimensione vocabolario: {get_vocab_size():,}")
        print(f"  - Contesto totale: {2 * CONTEXT_WINDOW} token")
        
        # Carica dataset in modalità LAZY
        print(f"\n📂 Test 3: Caricamento dataset 1900s (modalità LAZY)...")
        dataset = CBOWDataset(decade="1900s", cache_contexts=False)
        
        print(f"✓ Dataset caricato")
        print(f"  - Contesti stimati: {len(dataset):,}")
        
        if len(dataset) > 0:
            # Accedi a alcuni contesti on-demand
            print(f"\n🔍 Test 4: Accesso a contesti on-demand:")
            
            # Test indice 0
            target, context = dataset[0]
            print(f"  - Contesto 0:")
            print(f"    Target index: {target.item()}")
            print(f"    Context shape: {context.shape}")
            
            # Test indice casuale
            random_idx = len(dataset) // 2
            target, context = dataset[random_idx]
            print(f"  - Contesto {random_idx}:")
            print(f"    Target index: {target.item()}")
            print(f"    Context shape: {context.shape}")
            
            # Decodifica
            target_word = reverse_vocab.get(target.item(), "<UNKNOWN>")
            context_words = [reverse_vocab.get(int(idx), "<UNKNOWN>") for idx in context.tolist()[:3]]
            print(f"  - Target parola: '{target_word}'")
            print(f"  - Contesto parole (primi 3): {context_words}")
            
            # Test DataLoader con shuffle
            print(f"\n🔄 Test 5: PyTorch DataLoader (LAZY mode)")
            from torch.utils.data import DataLoader
            
            loader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=False,
                num_workers=0
            )
            
            # Primo batch
            batch_targets, batch_contexts = next(iter(loader))
            print(f"✓ Batch caricato:")
            print(f"  - Batch targets shape: {batch_targets.shape}")
            print(f"  - Batch contexts shape: {batch_contexts.shape}")
            print(f"  - Batch targets (primi 5): {batch_targets[:5].tolist()}")
            
            # Misura memoria approssimativa
            import sys
            estimated_memory_mb = len(dataset) * (2 * CONTEXT_WINDOW + 1) * 8 / (1024*1024)  # 8 bytes per int64
            print(f"\n💾 Memoria stimata per dataset completo:")
            print(f"  - Se cachato: ~{estimated_memory_mb:.0f} MB")
            print(f"  - In LAZY mode: <10 MB (streaming)")
        
        print(f"\n✅ Test completato con successo!")
        
    except Exception as e:
        import traceback
        print(f"\n❌ Errore durante il test:")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
