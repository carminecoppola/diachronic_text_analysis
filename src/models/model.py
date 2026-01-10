"""
FASE 5a: Modello CBOW (Continuous Bag of Words)

Architettura:
1. Input: contesto (batch_size, 2*CONTEXT_WINDOW)
2. Embedding layer: converti indici → vettori densi (batch_size, 2*CONTEXT_WINDOW, embedding_dim)
3. Average pooling: media del contesto (batch_size, embedding_dim)
4. Output layer: predici il target word (batch_size, vocab_size)
5. Loss: CrossEntropyLoss

La rete impara embeddings tali che:
  - Parole con contesti simili hanno vettori simili
  - Parole correlate semanticamente si raggruppano nello spazio

Questo è il modello base; la "magia" del semantic drift avviene quando
confrontiamo questi embeddings tra decenni diversi.
"""

import torch
import torch.nn as nn
from src.config import EMBEDDING_DIM, VOCAB_SIZE


class CBOWModel(nn.Module):
    """
    Modello CBOW (Continuous Bag of Words) per word embeddings.
    
    Predice una parola target dato il contesto circostante.
    
    Attributes:
        vocab_size (int): Dimensione vocabolario (50,001)
        embedding_dim (int): Dimensione vettori embedding (300)
        embeddings (nn.Embedding): Layer embedding del contesto
        output_layer (nn.Linear): Layer di output (predizione target)
    """
    
    def __init__(self, vocab_size: int = VOCAB_SIZE, embedding_dim: int = EMBEDDING_DIM):
        """
        Inizializza il modello CBOW.
        
        Args:
            vocab_size (int): Numero di token nel vocabolario
            embedding_dim (int): Dimensione degli embedding
        """
        super(CBOWModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Layer embedding per il contesto
        # Mappa indici token → vettori densi
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # Indice <UNK> e <PAD> non contribuiscono ai gradienti
        )
        
        # Layer di output
        # Mappa embedding mediato → logits per ogni parola (vocab_size dimensioni)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        # Inizializzazione pesi
        self._init_weights()
        
        print(f"✓ Modello CBOW creato:")
        print(f"  - Vocabolario: {vocab_size:,} token")
        print(f"  - Dimensione embedding: {embedding_dim}")
        print(f"  - Parametri totali: {self._count_parameters():,}")
    
    def _init_weights(self):
        """Inizializzazione pesi per convergenza più veloce."""
        # Embedding: distribuzione uniforme ristretta
        self.embeddings.weight.data.uniform_(-0.5 / self.embedding_dim, 0.5 / self.embedding_dim)
        
        # Output layer: inizializzazione Xavier
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def _count_parameters(self) -> int:
        """Conta i parametri allenabili."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, context_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modello CBOW.
        
        Args:
            context_indices (torch.Tensor): Indici contesto
                Shape: [batch_size, 2*CONTEXT_WINDOW]
                Valori: indici token nel vocabolario (0-50000)
        
        Returns:
            torch.Tensor: Logits predetti per il target word
                Shape: [batch_size, vocab_size]
                Valori: score non normalizzati per ogni token
        
        Process:
            1. Embedding: [batch, 2*cw] → [batch, 2*cw, emb_dim]
            2. Average pooling: [batch, 2*cw, emb_dim] → [batch, emb_dim]
            3. Output layer: [batch, emb_dim] → [batch, vocab_size]
        """
        # Step 1: Converti indici in embedding
        # context_indices shape: [batch_size, 2*CONTEXT_WINDOW]
        embedded = self.embeddings(context_indices)
        # embedded shape: [batch_size, 2*CONTEXT_WINDOW, embedding_dim]
        
        # Step 2: Average pooling sul contesto
        # Calcola la media degli embedding del contesto
        context_embedding = torch.mean(embedded, dim=1)
        # context_embedding shape: [batch_size, embedding_dim]
        
        # Step 3: Predizioni (logits)
        output = self.output_layer(context_embedding)
        # output shape: [batch_size, vocab_size]
        
        return output
    
    def get_embedding_weights(self) -> torch.Tensor:
        """
        Restituisce i pesi degli embedding (matrice parola→vettore).
        
        Utile per:
        - Estrarre i vettori finali delle parole
        - Calcolare similarità tra parole
        - Visualizzare con PCA/t-SNE
        
        Returns:
            torch.Tensor: Matrice embedding [vocab_size, embedding_dim]
        """
        return self.embeddings.weight.detach()
    
    def get_word_vector(self, word_idx: int) -> torch.Tensor:
        """
        Ottiene il vettore embedding per una parola specifica.
        
        Args:
            word_idx (int): Indice della parola nel vocabolario
        
        Returns:
            torch.Tensor: Vettore embedding [embedding_dim]
        """
        return self.embeddings.weight[word_idx].detach()


# ============================================================================
# TEST & DEBUG
# ============================================================================

if __name__ == "__main__":
    """
    Test del modello CBOW
    """
    print("🧪 Test Modello CBOW\n")
    
    # Test 1: Creazione modello
    print("Test 1: Creazione modello...")
    model = CBOWModel(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM)
    print()
    
    # Test 2: Forward pass
    print("Test 2: Forward pass...")
    batch_size = 32
    context_window = 5
    context_size = 2 * context_window  # 10
    
    # Crea batch di esempio (indici casuali nel vocabolario)
    context_batch = torch.randint(0, VOCAB_SIZE, (batch_size, context_size))
    print(f"  - Input shape: {context_batch.shape} (batch_size={batch_size}, context_size={context_size})")
    
    # Forward pass
    output = model(context_batch)
    print(f"  - Output shape: {output.shape}")
    print(f"  - Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test 3: Loss function
    print(f"\nTest 3: Loss function...")
    targets = torch.randint(0, VOCAB_SIZE, (batch_size,))
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, targets)
    print(f"  - Target indices: {targets[:5].tolist()}... (batch_size={batch_size})")
    print(f"  - Loss: {loss.item():.4f}")
    
    # Test 4: Backward pass
    print(f"\nTest 4: Backward pass...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss.backward()
    optimizer.step()
    print(f"  - Gradients calcolati ✓")
    print(f"  - Ottimizzatore aggiornato ✓")
    
    # Test 5: Embedding extraction
    print(f"\nTest 5: Estrazione embedding...")
    word_idx = 1  # "the"
    word_vec = model.get_word_vector(word_idx)
    print(f"  - Vettore parola (indice {word_idx}): shape {word_vec.shape}")
    print(f"  - Primi 5 valori: {word_vec[:5].tolist()}")
    
    all_embeddings = model.get_embedding_weights()
    print(f"  - Matrice embedding completa: {all_embeddings.shape}")
    
    # Test 6: Device compatibility
    print(f"\nTest 6: Device compatibility...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    context_batch = context_batch.to(device)
    output = model(context_batch)
    print(f"  - Modello spostato su: {device} ✓")
    print(f"  - Output shape: {output.shape}")
    
    print(f"\n✅ Tutti i test passati!")
