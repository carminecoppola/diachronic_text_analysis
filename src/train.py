"""
FASE 5b: Training Loop per Modelli CBOW
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import glob

from config import (
    DECADES, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    EMBEDDING_DIM, VOCAB_SIZE, MODELS_DIR, get_model_path,
)
from dataset import CBOWDataset
from model import CBOWModel

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