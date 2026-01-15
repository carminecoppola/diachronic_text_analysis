"""
FASE 6: Ispezione e Visualizzazione Embeddings

Questo script permette di caricare un modello allenato (.pt) e interrogarlo.
Usa la Cosine Similarity per trovare parole simili e t-SNE per graficare.

USO:
    python src/inspect_embeddings.py --decade 1950s --word war
"""

import os
import sys

# --- AGGIUNGI QUESTO BLOCCO ---
# Ottieni il percorso assoluto della root del progetto
current_dir = os.path.dirname(os.path.abspath(__file__))
# Sale di due livelli: src/utils -> src -> ROOT
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)
# ------------------------------

import torch
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Import configurazioni locali
from src.config import MODELS_DIR, VOCAB_FILE, EMBEDDING_DIM, get_model_path

class EmbeddingInspector:
    def __init__(self, decade, device="cpu"):
        self.decade = decade
        self.device = device
        self.word2idx = {}
        self.idx2word = {}
        self.embeddings = None
        
        self._load_vocab()
        self._load_model()

    def _load_vocab(self):
        print(f"📖 Caricamento vocabolario da {VOCAB_FILE}...")
        with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
            self.word2idx = json.load(f)
        # Crea mappa inversa (Index -> Word)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        print(f"   Vocabolario size: {len(self.word2idx)}")

    def _load_model(self):
        model_path = get_model_path(self.decade)
        print(f"🧠 Caricamento modello: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ Errore: File modello non trovato!")
            sys.exit(1)

        # Carica i pesi (state_dict)
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Estrae solo la matrice degli embeddings
        # NOTA: Il nome della chiave dipende da come hai chiamato il layer in model.py
        # Solitamente è 'embeddings.weight' o 'embedding.weight'
        keys = list(state_dict.keys())
        emb_key = [k for k in keys if 'weight' in k][0] # Prende la prima chiave con 'weight'
        
        print(f"   Chiave trovata nel modello: {emb_key}")
        
        self.embeddings = state_dict[emb_key].cpu().numpy()
        print(f"   Matrice Shape: {self.embeddings.shape}")

    def get_vector(self, word):
        """Restituisce il vettore per una parola."""
        idx = self.word2idx.get(word.lower())
        if idx is None:
            return None
        return self.embeddings[idx]

    def find_similar(self, word, top_k=10):
        """Trova le k parole più simili (Cosine Similarity)."""
        idx = self.word2idx.get(word.lower())
        if idx is None:
            print(f"⚠️  Parola '{word}' non trovata nel vocabolario.")
            return []

        # Vettore target
        target_vec = self.embeddings[idx]
        
        # Calcolo Cosine Similarity vettoriale (Matrix multiplication)
        # Sim = (A . B) / (||A|| * ||B||)
        
        # Norm delle righe della matrice
        norm_matrix = np.linalg.norm(self.embeddings, axis=1)
        norm_target = np.linalg.norm(target_vec)
        
        # Dot product
        dot_products = np.dot(self.embeddings, target_vec)
        
        # Similarità
        cosine_sims = dot_products / (norm_matrix * norm_target + 1e-9)
        
        # Ordina indici (dal più simile al meno simile)
        # argsort ordina crescente, quindi prendiamo gli ultimi
        sorted_indices = np.argsort(cosine_sims)[::-1]
        
        results = []
        # Saltiamo lo 0 perché è la parola stessa
        for i in range(1, top_k + 1):
            curr_idx = sorted_indices[i]
            similarity = cosine_sims[curr_idx]
            word_res = self.idx2word.get(curr_idx, "<UNK>")
            results.append((word_res, similarity))
            
        return results

    def plot_tsne(self, words_to_plot):
        """Visualizza un sottoinsieme di parole con t-SNE."""
        print("\n🎨 Generazione grafico t-SNE...")
        
        # Filtra parole esistenti
        valid_words = [w for w in words_to_plot if w in self.word2idx]
        indices = [self.word2idx[w] for w in valid_words]
        
        if not indices:
            print("Nessuna parola valida da graficare.")
            return

        vectors = self.embeddings[indices]

        # Riduci a 2 dimensioni
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(vectors)-1))
        vectors_2d = tsne.fit_transform(vectors)

        # Plot
        plt.figure(figsize=(12, 8))
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='red', edgecolors='k')

        for i, word in enumerate(valid_words):
            plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]), 
                         xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

        plt.title(f"Visualizzazione Embeddings - Decade {self.decade}")
        plt.grid(True, alpha=0.3)
        
        filename = f"tsne_{self.decade}.png"
        plt.savefig(filename)
        print(f"✅ Grafico salvato come: {filename}")

# ==================================================================
# MAIN
# ==================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decade", type=str, default="1950s", help="Decade da ispezionare")
    parser.add_argument("--word", type=str, default="war", help="Parola di cui cercare simili")
    args = parser.parse_args()

    inspector = EmbeddingInspector(args.decade)

    # 1. Test Similarità
    print(f"\n🔍 Vicini semantici di '{args.word}':")
    similar = inspector.find_similar(args.word)
    
    if similar:
        print(f"{'Parola':<20} | {'Similarità':<10}")
        print("-" * 35)
        for w, s in similar:
            print(f"{w:<20} | {s:.4f}")
    
    # 2. Test Visualizzazione (Lista Curata)
    curated_list = [
        "war", "peace", "army", "battle", "soldier",
        "man", "woman", "boy", "girl", "father", "mother",
        "king", "queen", "president", "government",
        "money", "gold", "market", "business",
        "science", "machine", "engine", "radio", "computer",
        "good", "bad", "happy", "sad"
    ]
    
    # Aggiungiamo anche i simili trovati prima al plot
    plot_words = list(set(curated_list + [w for w, s in similar] + [args.word]))
    
    inspector.plot_tsne(plot_words)