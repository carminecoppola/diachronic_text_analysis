import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import CBOWDataset
from src.models.model import CBOWModel
from src.config import EMBEDDING_DIM

torch.manual_seed(0)

def run_one_step(model, batch, device="cpu"):
    model.train()
    model.to(device)
    targets, contexts = batch
    targets = targets.to(device)
    contexts = contexts.to(device)

    opt = torch.optim.SGD(model.parameters(), lr=0.05)

    opt.zero_grad(set_to_none=True)
    logits = model(contexts)              # [B, V]
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    opt.step()
    return float(loss.item())

def main():
    d = CBOWDataset("1900s")
    dl = DataLoader(d, batch_size=256, shuffle=False)

    batch = next(iter(dl))

    # modello base iniziale identico per entrambi
    base_init = CBOWModel(
        vocab_size=len(d.word2idx),
        embedding_dim=EMBEDDING_DIM,
        stopword_mask=d.stopword_mask,
        stopword_keep_prob=d.stopword_keep_prob,
    )

    # clona due copie IDENTICHE
    m_baseline = copy.deepcopy(base_init)
    m_weighted = copy.deepcopy(base_init)

    # FORZA: baseline = niente weighting/subsampling
    m_baseline.use_stopword_weighting = False
    m_baseline.use_stopword_subsampling = False

    # FORZA: weighted = weighting on, subsampling off
    m_weighted.use_stopword_weighting = True
    m_weighted.use_stopword_subsampling = False
    m_weighted.stopword_weight = 0.2

    device = "cuda" if torch.cuda.is_available() else "cpu"

    loss0 = run_one_step(m_baseline, batch, device=device)
    loss1 = run_one_step(m_weighted, batch, device=device)

    # confronta embedding matrix dopo 1 step
    E0 = m_baseline.embeddings.weight.detach().cpu().float()
    E1 = m_weighted.embeddings.weight.detach().cpu().float()

    per_token = (E0 - E1).norm(dim=1)
    print("loss baseline:", loss0)
    print("loss weighted:", loss1)
    print("L2 diff mean after 1 step:", float(per_token.mean()))
    print("L2 diff max  after 1 step:", float(per_token.max()))

if __name__ == "__main__":
    main()
