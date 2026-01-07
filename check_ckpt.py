import torch
import os

# Il percorso del tuo file
ckpt_path = "/home/vbucciero/diachronic_text_analysis/checkpoints/1900s/ckpt_ep0_step160000.pt"

if not os.path.exists(ckpt_path):
    print("❌ File non trovato!")
    exit()

# Carica sulla CPU (così non serve la GPU per controllare)
checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

print(f"\n📦 Contenuto del Checkpoint: {os.path.basename(ckpt_path)}")
print("="*50)

# 1. Vedi le chiavi disponibili
print(f"🔑 Chiavi nel dizionario: {list(checkpoint.keys())}")

# 2. Info sul Training
epoch = checkpoint.get('epoch', 'N/A')
step = checkpoint.get('global_step', 'N/A')
loss = checkpoint.get('loss', 'N/A')

print(f"\n📊 Stato Training:")
print(f"  - Epoca: {epoch}")
print(f"  - Batch/Step globale: {step}")
print(f"  - Loss salvata: {loss}")

# 3. Info sul Modello (Embeddings)
state_dict = checkpoint.get('model_state_dict', {})
if 'embeddings.weight' in state_dict:
    emb_shape = state_dict['embeddings.weight'].shape
    print(f"\n🧠 Dimensioni Embeddings: {emb_shape}")
    print(f"  - Vocabolario: {emb_shape[0]} parole")
    print(f"  - Dimensioni vettore: {emb_shape[1]}")
    
    # Opzionale: Vedere i primi valori (per capire se sono tutti uguali)
    first_vec = state_dict['embeddings.weight'][0]
    print(f"  - Esempio primo vettore (primi 5 val): {first_vec[:5].tolist()}")
else:
    print("\n⚠️  Impossibile trovare 'embeddings.weight' nel modello.")

print("="*50)