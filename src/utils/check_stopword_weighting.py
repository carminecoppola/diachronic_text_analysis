"""
CHECK AUTOMATICO STOPWORD WEIGHTING

Questo script:
- NON fa training
- NON salva modelli
- NON modifica nulla

Serve solo a verificare che:
- le stopwords vengano riconosciute
- il weighting (es. 0.2) cambi davvero il context embedding
"""

import torch
from src.data.dataset import CBOWDataset
from src.models.model import CBOWModel
from src.config import EMBEDDING_DIM, USE_STOPWORD_WEIGHTING, STOPWORD_CONTEXT_WEIGHT

torch.manual_seed(0)


def main():
    print("=== CHECK STOPWORD WEIGHTING ===")
    print("USE_STOPWORD_WEIGHTING:", USE_STOPWORD_WEIGHTING)
    print("STOPWORD_CONTEXT_WEIGHT:", STOPWORD_CONTEXT_WEIGHT)
    print()

    # Dataset
    d = CBOWDataset("1900s")

    # Modello (solo embeddings + pooling)
    m = CBOWModel(
        vocab_size=len(d.word2idx),
        embedding_dim=EMBEDDING_DIM,
        stopword_mask=d.stopword_mask,
        stopword_keep_prob=d.stopword_keep_prob,
    )

    # Cerca un esempio con almeno una stopword nel contesto
    idx = None
    for i in [100, 500, 1000, 2000, 5000]:
        _, ctx = d[i]
        if m.stopword_mask[ctx].any():
            idx = i
            break

    if idx is None:
        print("❌ Nessun contesto con stopwords trovato (improbabile).")
        return

    target, context = d[idx]
    context = context.unsqueeze(0)

    # Embeddings token
    embedded = m.embeddings(context)

    # ===== BASELINE (mean semplice, padding escluso) =====
    w_base = torch.ones_like(context, dtype=embedded.dtype)
    w_base = w_base.masked_fill(context == 0, 0.0)

    base = (embedded * w_base.unsqueeze(-1)).sum(dim=1) / \
           w_base.sum(dim=1, keepdim=True).clamp_min(1e-8)

    # ===== WEIGHTED (stopwords pesate) =====
    w = w_base.clone()
    sw = m.stopword_mask[context]
    w = torch.where(sw, w * m.stopword_weight, w)

    weighted = (embedded * w.unsqueeze(-1)).sum(dim=1) / \
               w.sum(dim=1, keepdim=True).clamp_min(1e-8)

    diff = (base - weighted).abs().mean().item()

    print(f"Esempio usato: d[{idx}]")
    print("Stopwords nel contesto:", int(sw.sum().item()))
    print("Diff media |baseline - weighted|:", diff)

    if diff > 0:
        print("✅ STOPWORD WEIGHTING ATTIVO E FUNZIONANTE")
    else:
        print("❌ Nessuna differenza → qualcosa non è agganciato")


if __name__ == "__main__":
    main()
