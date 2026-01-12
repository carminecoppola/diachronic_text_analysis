"""
ONE-STEP CHECK (STOPWORDS-RICH BATCH) + CONTEXT-EMBEDDING DIFF

Cosa fa:
1) Carica dataset 1900s + modello CBOW (stesso init per due copie).
2) Scansiona i primi MAX_BATCHES batch e sceglie quello con più stopwords nel contesto.
3) Misura la differenza *diretta* sul context embedding:
      baseline (mean, padding escluso)
      vs
      weighted (stopwords pesate a stopword_weight)
   -> Questa è la misura più "pulita" perché è esattamente dove agisce la patch.
4) (Opzionale) Fa un solo step SGD su quel batch per baseline vs weighted,
   e misura quanto differiscono gli embedding dopo 1 step.

Nota:
- NON salva nulla
- NON allena per epoche intere
"""

import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import CBOWDataset
from src.models.model import CBOWModel
from src.config import EMBEDDING_DIM

torch.manual_seed(0)


def run_one_step(model, batch, device):
    model.train()
    model.to(device)

    targets, contexts = batch
    targets = targets.to(device)
    contexts = contexts.to(device)

    opt = torch.optim.SGD(model.parameters(), lr=0.05)

    opt.zero_grad(set_to_none=True)
    logits = model(contexts)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    opt.step()

    return float(loss.item())


def compute_context_embedding_diff(model, batch, stopword_weight=0.2):
    """
    Misura quanto cambia il *context embedding* (vettore [B, D]) quando:
    - baseline: pesi=1 (padding escluso)
    - weighted: stopwords pesate a stopword_weight

    Ritorna:
    - diff media assoluta tra i due context embedding
    - conteggio stopwords nel batch
    - token totali nel batch
    """
    _, contexts = batch  # [B, C]
    embedded = model.embeddings(contexts)  # [B, C, D]

    # baseline weights: 1 per token, 0 per padding/unk id=0
    w_base = torch.ones_like(contexts, dtype=embedded.dtype)
    w_base = w_base.masked_fill(contexts == 0, 0.0)

    base_ctx = (embedded * w_base.unsqueeze(-1)).sum(dim=1) / w_base.sum(dim=1, keepdim=True).clamp_min(1e-8)

    # weighted: stopwords * stopword_weight
    sw = model.stopword_mask[contexts]  # [B, C] bool
    w = torch.where(sw, w_base * float(stopword_weight), w_base)

    weighted_ctx = (embedded * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1, keepdim=True).clamp_min(1e-8)

    ctx_diff = (base_ctx - weighted_ctx).abs().mean().item()
    sw_count = int(sw.sum().item())
    total_tokens = int(contexts.numel())

    return ctx_diff, sw_count, total_tokens


def main():
    print("=== STOPWORDS WEIGHTING CHECK (RICH BATCH) ===")

    # -------- dataset + dataloader
    d = CBOWDataset("1900s")
    dl = DataLoader(d, batch_size=512, shuffle=False)

    # -------- modello base (serve per embeddings + stopword_mask)
    base_init = CBOWModel(
        vocab_size=len(d.word2idx),
        embedding_dim=EMBEDDING_DIM,
        stopword_mask=d.stopword_mask,
        stopword_keep_prob=d.stopword_keep_prob,
    )

    # -------- STEP 1: trova batch con molte stopwords
    print("Scansione batch per trovare contesto ricco di stopwords...")

    best_batch = None
    best_sw = -1

    MAX_BATCHES = 200
    for i, batch in enumerate(dl):
        if i >= MAX_BATCHES:
            break

        _, contexts = batch
        sw_count = int(base_init.stopword_mask[contexts].sum().item())

        if sw_count > best_sw:
            best_sw = sw_count
            best_batch = batch

    print(f"✔ Batch selezionato: stopwords totali nel batch = {best_sw}")
    if best_batch is None or best_sw <= 0:
        print("❌ Nessun batch con stopwords trovato (improbabile).")
        return

    # -------- STEP 2: misura DIRECT sul context embedding (qui agisce la patch)
    STOPWORD_WEIGHT = 0.2
    ctx_diff, sw_count, total_tokens = compute_context_embedding_diff(
        base_init, best_batch, stopword_weight=STOPWORD_WEIGHT
    )

    print()
    print("---- DIFF SUL CONTEXT EMBEDDING (FORWARD) ----")
    print("stopword_weight:", STOPWORD_WEIGHT)
    print("Stopwords nel batch:", sw_count)
    print("Token totali nel batch:", total_tokens)
    print("Stopword ratio:", sw_count / max(1, total_tokens))
    print("Context embedding avg_abs_diff (baseline vs weighted):", ctx_diff)

    # -------- STEP 3: (opzionale) one-step update e confronto embedding
    print()
    print("---- ONE-STEP SGD (OPZIONALE) ----")

    m_baseline = copy.deepcopy(base_init)
    m_weighted = copy.deepcopy(base_init)

    # baseline: NO weighting/subsampling
    m_baseline.use_stopword_weighting = False
    m_baseline.use_stopword_subsampling = False

    # weighted: SI weighting, NO subsampling
    m_weighted.use_stopword_weighting = True
    m_weighted.use_stopword_subsampling = False
    m_weighted.stopword_weight = STOPWORD_WEIGHT

    device = "cuda" if torch.cuda.is_available() else "cpu"

    loss_base = run_one_step(m_baseline, best_batch, device)
    loss_weight = run_one_step(m_weighted, best_batch, device)

    E0 = m_baseline.embeddings.weight.detach().cpu().float()
    E1 = m_weighted.embeddings.weight.detach().cpu().float()
    diff = (E0 - E1).norm(dim=1)

    print("loss baseline :", loss_base)
    print("loss weighted :", loss_weight)
    print("L2 diff mean after 1 step:", float(diff.mean()))
    print("L2 diff max  after 1 step:", float(diff.max()))
    print("✅ CHECK COMPLETATO")


if __name__ == "__main__":
    main()
