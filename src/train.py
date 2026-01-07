"""
FASE 5b: Training Loop per Modelli CBOW

Questo script allena un modello CBOW separato per ogni decennio (1900s-1990s).

WORKFLOW:
1. Per ogni decennio:
   a. Carica dataset CBOW (contesti dal file processato)
   b. Inizializza modello
   c. Esegue N epoch di training
   d. Salva embeddings in models/emb_XXXX.pt

ARCHITETTURA:
- Input batch: contesti CBOW (batch_size, 2*CONTEXT_WINDOW)
- Predizione: parola target
- Loss: CrossEntropyLoss
- Optimizer: Adam (con learning rate decay)

OUTPUT:
- models/emb_1900s.pt, emb_1910s.pt, ..., emb_1990s.pt
- Ogni file contiene i pesi dell'embedding layer
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

from config import (
    DECADES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    EMBEDDING_DIM,
    VOCAB_SIZE,
    MODELS_DIR,
    get_model_path,
)
from dataset import CBOWDataset
from model import CBOWModel


class Trainer:
    """
    Trainer per modelli CBOW.
    
    Gestisce:
    - Caricamento dataset
    - Loop di training
    - Salvataggio modelli
    - Logging e metriche
    """
    
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embedding_dim: int = EMBEDDING_DIM,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        epochs: int = EPOCHS, # SUGGERIMENTO: Imposta a 1 se il dataset è > 1Mld token
        device: str = None,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mixed Precision Scaler (Velocizza su GPU V100/A100/RTX)
        self.scaler = torch.cuda.amp.GradScaler() 
        
        print(f"🚀 Trainer inizializzato su {self.device}")

    def _get_latest_checkpoint(self, decade):
        """Cerca il checkpoint più recente per il decennio."""
        ckpt_dir = os.path.join("checkpoints", decade)
        if not os.path.exists(ckpt_dir):
            return None
        
        list_of_files = glob.glob(f"{ckpt_dir}/*.pt")
        if not list_of_files:
            return None
            
        # Trova il file più recente
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file

    def train_decade(self, decade: str) -> dict:
        print(f"\n{'='*60}")
        print(f"📚 Training: {decade}")
        print(f"{'='*60}")

        # 1. SETUP DATASET
        # -------------------------------------------------------
        # NOTA: Su HPC Linux, num_workers DEVE essere > 0 (es. 4 o 8)
        # Se usi 0, la GPU morirà di noia aspettando la CPU.
        WORKERS = 4 if self.device == "cuda" else 0
        
        try:
            dataset = CBOWDataset(decade=decade, cache_contexts=False)
        except Exception as e:
            print(f"❌ Errore dataset: {e}")
            return None

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=WORKERS, 
            pin_memory=True,
            prefetch_factor=2 if WORKERS > 0 else None
        )
        num_batches = len(dataloader)

        # 2. SETUP MODELLO & OPTIMIZER
        # -------------------------------------------------------
        model = CBOWModel(self.vocab_size, self.embedding_dim).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # 3. GESTIONE RESUME (CHECKPOINTING)
        # -------------------------------------------------------
        start_epoch = 0
        global_step = 0
        latest_ckpt = self._get_latest_checkpoint(decade)
        
        if latest_ckpt:
            print(f"♻️  Trovato checkpoint: {latest_ckpt}")
            print("   Caricamento pesi e ripresa training...")
            checkpoint = torch.load(latest_ckpt, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Opzionale: caricare lo scaler se vuoi precisione perfetta al riavvio
            # self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch']
            global_step = checkpoint.get('global_step', 0)
            print(f"   Ripresa dall'epoch {start_epoch}, step globale {global_step}")
        else:
            print("🆕 Nessun checkpoint trovato. Inizio da zero.")

        # 4. TRAINING LOOP
        # -------------------------------------------------------
        model.train()
        SAVE_EVERY_STEPS = 10000  # Salva ogni 10k batch (circa ogni 15-30 min)
        
        # Directory per i checkpoint parziali
        ckpt_dir = os.path.join("checkpoints", decade)
        os.makedirs(ckpt_dir, exist_ok=True)

        for epoch in range(start_epoch, self.epochs):
            total_loss = 0
            pbar = tqdm(dataloader, desc=f"Ep {epoch+1}/{self.epochs}", initial=global_step % num_batches, total=num_batches)
            
            for batch_idx, (targets, contexts) in enumerate(pbar):
                # Se abbiamo ripreso, saltiamo i batch (simulato per semplicità, 
                # in realtà il dataloader reshuffla, ma va bene così per embedding massivi)
                # Una logica più complessa richiederebbe un sampler deterministico.
                
                targets = targets.to(self.device)
                contexts = contexts.to(self.device)

                optimizer.zero_grad()

                # MIXED PRECISION (Velocità 2x su GPU moderna)
                with torch.cuda.amp.autocast(enabled=(self.device=="cuda")):
                    outputs = model(contexts)
                    loss = criterion(outputs, targets)

                # Scaled Backward
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                # Metrics
                current_loss = loss.item()
                total_loss += current_loss
                global_step += 1
                
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})

                # SALVATAGGIO INTERMEDIO (CHECKPOINT)
                if global_step % SAVE_EVERY_STEPS == 0:
                    ckpt_name = os.path.join(ckpt_dir, f"ckpt_ep{epoch}_step{global_step}.pt")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': current_loss,
                    }, ckpt_name)
                    # Rimuovi vecchi checkpoint per risparmiare spazio (opzionale)
                    # ...

            # Fine Epoca
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch+1} completata. Avg Loss: {avg_loss:.4f}")
            
            # Salva checkpoint di fine epoca
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(ckpt_dir, f"epoch_{epoch+1}_done.pt"))

        # 5. SALVATAGGIO FINALE (MODELLO PULITO)
        # -------------------------------------------------------
        print(f"\n💾 Salvataggio modello finale...")
        final_path = get_model_path(decade)
        torch.save(model.state_dict(), final_path)
        
        # Pulizia checkpoint (opzionale: cancella la cartella checkpoints/decade se finito)
        # shutil.rmtree(ckpt_dir) 

        return {
            "decade": decade,
            "final_loss": avg_loss if 'avg_loss' in locals() else 0.0,
            "training_time": "N/A"
        }
    
    def train_all_decades(self) -> dict:
        """
        Allena modelli per TUTTI i decenni.
        
        Returns:
            Dict con metriche di tutti i decenni
        """
        print(f"\n{'#'*70}")
        print(f"# 🎯 TRAINING COMPLETO: TUTTI I DECENNI (1900s-1990s)")
        print(f"{'#'*70}")
        
        all_metrics = {}
        failed_decades = []
        
        for decade in DECADES:
            try:
                metrics = self.train_decade(decade)
                if metrics:
                    all_metrics[decade] = metrics
                else:
                    failed_decades.append(decade)
            except Exception as e:
                print(f"❌ Errore durante training {decade}: {e}")
                failed_decades.append(decade)
        
        # Sommario finale
        print(f"\n{'='*70}")
        print(f"📊 SOMMARIO TRAINING")
        print(f"{'='*70}")
        
        print(f"\n✅ Decenni completati: {len(all_metrics)}/{len(DECADES)}")
        for decade, metrics in all_metrics.items():
            print(f"  ✓ {decade}: Loss={metrics['final_loss']:.4f} ({metrics['training_time']})")
        
        if failed_decades:
            print(f"\n❌ Decenni falliti: {failed_decades}")
        
        # Salva sommario
        summary_path = os.path.join(MODELS_DIR, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n📝 Sommario salvato: {summary_path}")
        
        return all_metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Entry point dello script di training."""
    
    # Verifica directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Crea trainer
    trainer = Trainer(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
    )
    
    # Allena tutti i decenni
    metrics = trainer.train_all_decades()
    
    # Stato finale
    if len(metrics) == len(DECADES):
        print(f"\n{'🎉'*35}")
        print(f"✅ TRAINING COMPLETATO CON SUCCESSO!")
        print(f"{'🎉'*35}")
        print(f"\nModelli salvati in: {MODELS_DIR}")
        print(f"  - emb_1900s.pt")
        print(f"  - emb_1910s.pt")
        print(f"  - ...")
        print(f"  - emb_1990s.pt")
        return 0
    else:
        print(f"\n⚠️  Training completato con errori!")
        print(f"Decenni completati: {len(metrics)}/{len(DECADES)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
