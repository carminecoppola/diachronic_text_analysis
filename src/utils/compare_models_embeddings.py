import torch
from src.config import VOCAB_FILE

# cambia questi path con quelli reali dei due finali
BASELINE = "models_compare_baseline/5gram/emb_1900s.pt"
WEIGHTED = "models_compare_weighted/5gram/emb_1900s.pt"

def load_emb(path):
    obj = torch.load(path, map_location="cpu")
    # supporta sia dict che tensor
    if isinstance(obj, dict):
        # prova chiavi comuni
        for k in ["embeddings", "embedding", "weight", "state_dict"]:
            if k in obj:
                v = obj[k]
                if isinstance(v, torch.Tensor):
                    return v
                if isinstance(v, dict):
                    # cerca embedding weight dentro state_dict
                    for kk in v.keys():
                        if "emb" in kk and "weight" in kk:
                            return v[kk]
        # fallback: se è uno state_dict puro
        for kk in obj.keys():
            if "emb" in kk and "weight" in kk:
                return obj[kk]
        raise KeyError(f"Non trovo embedding in {path}. Chiavi: {list(obj.keys())[:20]}")
    elif isinstance(obj, torch.Tensor):
        return obj
    else:
        raise TypeError(type(obj))

def main():
    E0 = load_emb(BASELINE).float()
    E1 = load_emb(WEIGHTED).float()

    assert E0.shape == E1.shape, (E0.shape, E1.shape)

    # differenza media per token
    per_token = (E0 - E1).norm(dim=1)      # [V]
    print("Embedding shape:", tuple(E0.shape))
    print("L2 diff mean:", float(per_token.mean()))
    print("L2 diff median:", float(per_token.median()))
    print("L2 diff max:", float(per_token.max()))

    # mostra i top-10 token più cambiati (per indice)
    topk = torch.topk(per_token, k=10)
    print("Top10 indices:", topk.indices.tolist())
    print("Top10 diffs:", [float(x) for x in topk.values])

if __name__ == "__main__":
    main()
