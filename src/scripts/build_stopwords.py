"""
ESEGUI UNA SOLA VOLTA.

Scopo:
- Prende la lista stopwords inglesi da NLTK
- La congela su file: data/stopwords_en.txt
Così il training non dipende da NLTK né da download runtime.
"""

import nltk
from nltk.corpus import stopwords
import os

nltk.download("stopwords")

"""os.makedirs("data", exist_ok=True)

out_path = os.path.join("data", "stopwords_en.txt")
with open(out_path, "w", encoding="utf-8") as f:
    for w in sorted(stopwords.words("english")):
        f.write(w + "\n")"""

# Scrive nella cartella corrente dello script: .../src/scripts
out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "stopwords_en.txt")

with open(out_path, "w", encoding="utf-8") as f:
    for w in sorted(stopwords.words("english")):
        f.write(w + "\n")

print(f"Creato {out_path}")
