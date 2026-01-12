import argparse
import json
import torch
import torch.nn.functional as F

BASELINE_PATH = "models_compare_baseline/5gram/emb_1900s.pt"
WEIGHTED_PATH = "models_compare_weighted_w05/5gram/emb_1900s.pt"
VOCAB_PATH = "data/processed/5gram/vocab.json"


def load_vocab_token2id(path: str):
    """
    Nel tuo progetto vocab.json è un dict diretto: token -> id (int).
    """
    token2id = json.load(open(path, "r", encoding="utf-8"))
    if not isinstance(token2id, dict):
        raise ValueError("vocab.json non è un dict token->id")

    vocab_size = max(token2id.values()) + 1
    idx2word = [""] * vocab_size
    for w, i in token2id.items():
        idx2word[i] = w

    return token2id, idx2word


def load_embedding_matrix(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        return obj.float()

    if isinstance(obj, dict):
        sd = obj["state_dict"] if ("state_dict" in obj and isinstance(obj["state_dict"], dict)) else obj
        best = None
        for _, v in sd.items():
            if isinstance(v, torch.Tensor) and v.dim() == 2 and v.size(0) > 1000:
                if best is None or v.size(0) > best.size(0):
                    best = v
        if best is None:
            raise KeyError("Non trovo embedding matrix 2D nel checkpoint.")
        return best.float()

    raise TypeError(type(obj))


def topk_neighbors(E: torch.Tensor, idx: int, k: int):
    En = F.normalize(E, dim=1)
    sims = torch.mv(En, En[idx])
    sims[idx] = -1.0
    vals, inds = torch.topk(sims, k=k)
    return inds.tolist(), vals.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", nargs="+", required=True, help="Parole query (spazio-separate)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--diffonly", action="store_true")
    args = ap.parse_args()

    token2id, idx2word = load_vocab_token2id(VOCAB_PATH)
    E_base = load_embedding_matrix(BASELINE_PATH)
    E_w = load_embedding_matrix(WEIGHTED_PATH)

    print("Embedding shape:", tuple(E_base.shape))
    print("Topk:", args.topk, "Diffonly:", args.diffonly)
    print("-" * 60)

    for w in args.words:
        if w not in token2id:
            print(f"'{w}' NON nel vocab\n")
            continue

        idx = token2id[w]
        ib, vb = topk_neighbors(E_base, idx, args.topk)
        iw, vw = topk_neighbors(E_w, idx, args.topk)

        nb = [(idx2word[j], s) for j, s in zip(ib, vb)]
        nw = [(idx2word[j], s) for j, s in zip(iw, vw)]

        set_b = {t for t, _ in nb}
        set_w = {t for t, _ in nw}
        overlap = len(set_b & set_w)

        print("=" * 80)
        print(f"WORD: '{w}' (idx={idx}) overlap top-{args.topk}: {overlap}/{args.topk}")

        if not args.diffonly:
            print("Baseline:")
            for t, s in nb:
                print(f"  {t:<22s} {s:.4f}")
            print("Weighted:")
            for t, s in nw:
                print(f"  {t:<22s} {s:.4f}")
        else:
            only_b = [(t, s) for t, s in nb if t not in set_w]
            only_w = [(t, s) for t, s in nw if t not in set_b]
            print("Only baseline:")
            for t, s in only_b:
                print(f"  {t:<22s} {s:.4f}")
            print("Only weighted:")
            for t, s in only_w:
                print(f"  {t:<22s} {s:.4f}")

        print()


if __name__ == "__main__":
    main()
