"""
FASE 5b: Training Loop per Modelli CBOW


import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import glob

from src.config import (
    DECADES, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    EMBEDDING_DIM, VOCAB_SIZE, MODELS_DIR, get_model_path,
)
from src.data.dataset import CBOWDataset
from src.models.model import CBOWModel

class Trainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = torch.cuda.amp.GradScaler()
        print(f"🚀 Trainer inizializzato su {self.device}")

    def train_decade(self, decade: str):
        print(f"\n{'='*60}\n📚 Training: {decade}\n{'='*60}")

        # 1. DATASET
        try:
            dataset = CBOWDataset(decade=decade)
        except Exception as e:
            print(f"❌ Errore dataset: {e}")
            return None
            
        if len(dataset) == 0:
            print("⚠️ Dataset vuoto o window troppo grande per i dati. Salto.")
            return None

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4 if self.device == "cuda" else 0,
            pin_memory=True
        )

        # 2. MODELLO
        actual_vocab_size = len(dataset.word2idx)
        model = CBOWModel(actual_vocab_size, EMBEDDING_DIM).to(self.device)
        criterion = nn.CrossEntropyLoss() # Nessun ignore_index necessario
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 3. LOOP
        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            pbar = tqdm(dataloader, desc=f"Ep {epoch+1}/{EPOCHS}")
            
            for targets, contexts in pbar:
                targets, contexts = targets.to(self.device), contexts.to(self.device)

                optimizer.zero_grad()
                
                # Mixed Precision
                with torch.cuda.amp.autocast(enabled=(self.device=="cuda")):
                    outputs = model(contexts)
                    loss = criterion(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch+1} completata. Avg Loss: {avg_loss:.4f}")

        # 4. SALVA
        print(f"💾 Salvataggio modello finale...")
        torch.save(model.state_dict(), get_model_path(decade))
        return {"decade": decade, "final_loss": avg_loss, "training_time": "N/A"}

    def train_all_decades(self):
        metrics = {}
        for decade in DECADES:
            res = self.train_decade(decade)
            if res: metrics[decade] = res
            
        # Salva sommario
        with open(os.path.join(MODELS_DIR, "training_summary.json"), 'w') as f:
            json.dump(metrics, f, indent=2)

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    trainer = Trainer()
    trainer.train_all_decades()

if __name__ == "__main__":
    sys.exit(main())
"""

"""
FASE 5b: Training Loop per Modelli CBOW (Versione Avanzata con Scheduler)
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
import json
import glob

# Importiamo config. Se SHOW_PROGRESS_BAR non c'è, mettiamo True di default
try:
    from src.config import SHOW_PROGRESS_BAR
except ImportError:
    SHOW_PROGRESS_BAR = True

from src.config import (
    DECADES, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    EMBEDDING_DIM, VOCAB_SIZE, MODELS_DIR, get_model_path,
)
from src.data.dataset import CBOWDataset
from src.models.model import CBOWModel

class Trainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = torch.amp.GradScaler('cuda')
        print(f"🚀 Trainer inizializzato su {self.device}")

    def train_decade(self, decade: str):
        print(f"\n{'='*60}\n📚 Training: {decade}\n{'='*60}")

        # 1. DATASET
        try:
            dataset = CBOWDataset(decade=decade)
        except Exception as e:
            print(f"❌ Errore dataset: {e}")
            return None
            
        if len(dataset) == 0:
            print("⚠️ Dataset vuoto o window troppo grande per i dati. Salto.")
            return None

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4 if self.device == "cuda" else 0,
            pin_memory=True
        )

        # 2. MODELLO
        actual_vocab_size = len(dataset.word2idx)
        model = CBOWModel(actual_vocab_size, EMBEDDING_DIM).to(self.device)
        criterion = nn.CrossEntropyLoss()
        
        # OTTIMIZZATORE & SCHEDULER (Novità!)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Scheduler Lineare: Parte da LR pieno e scende fino all'1% alla fine delle epoche
        # Questo stabilizza enormemente i vettori finali.
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=EPOCHS)

        # 3. LOOP
        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            
            # Gestione barra progresso intelligente (non intasa i log se disabilitata)
            pbar = tqdm(dataloader, desc=f"Ep {epoch+1}/{EPOCHS}", disable=not SHOW_PROGRESS_BAR)
            
            for targets, contexts in pbar:
                targets, contexts = targets.to(self.device), contexts.to(self.device)

                optimizer.zero_grad()
                
                # Mixed Precision
                with torch.cuda.amp.autocast(enabled=(self.device=="cuda")):
                    outputs = model(contexts)
                    loss = criterion(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                total_loss += loss.item()
                if SHOW_PROGRESS_BAR:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})

            # Step dello Scheduler alla fine dell'epoca
            scheduler.step()
            
            avg_loss = total_loss / len(dataloader)
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1} completata. Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

            # CHECKPOINTING (Salvataggio intermedio)
            # Salva una copia del modello ad ogni epoca. Utile se crasha o per analisi.
            epoch_path = os.path.join(MODELS_DIR, f"emb_{decade}_ep{epoch+1}.pt")
            torch.save(model.state_dict(), epoch_path)

        # 4. SALVA IL MODELLO FINALE (Sovrascrive o crea quello "pulito")
        print(f"💾 Salvataggio modello finale...")
        final_path = get_model_path(decade)
        torch.save(model.state_dict(), final_path)
        
        # Pulizia opzionale: rimuovi i checkpoint intermedi per risparmiare spazio?
        # Per ora li lasciamo, meglio averli.
        
        return {"decade": decade, "final_loss": avg_loss, "training_time": "N/A"}

    def train_all_decades(self):
        metrics = {}
        for decade in DECADES:
            res = self.train_decade(decade)
            if res: metrics[decade] = res
            
        # Salva sommario
        with open(os.path.join(MODELS_DIR, "training_summary.json"), 'w') as f:
            json.dump(metrics, f, indent=2)

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    trainer = Trainer()
    trainer.train_all_decades()

if __name__ == "__main__":
    sys.exit(main())