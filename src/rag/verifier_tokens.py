"""Module 3, étape A : vérifier que les morceaux respectent la limite du modèle."""
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

MODELE = "sentence-transformers/all-MiniLM-L6-v2"
FICHIER_CHUNKS = Path("data/processed/chunks.jsonl")

modele = SentenceTransformer(MODELE)
tokeniseur = modele.tokenizer
limite = modele.max_seq_length
print(f"Limite du modèle : {limite} tokens\n")

# 1. Voir comment le tokeniseur découpe une phrase technique
phrase = "MVSS-Net detects image splicing using JPEG compression artifacts."
print("Phrase :", phrase)
print("Tokens :", tokeniseur.tokenize(phrase), "\n")

# 2. Mesurer tous les morceaux
chunks = [json.loads(ligne) for ligne in FICHIER_CHUNKS.open(encoding="utf-8")]
nb_tokens = [len(tokeniseur.encode(c["texte"])) for c in chunks]
nb_caracteres = [len(c["texte"]) for c in chunks]

print(f"Morceaux analysés : {len(chunks)}")
print(f"Tokens par morceau : min {min(nb_tokens)}, moyenne {sum(nb_tokens) // len(nb_tokens)}, max {max(nb_tokens)}")
print(f"Caractères par token (moyenne) : {sum(nb_caracteres) / sum(nb_tokens):.2f}")
depassements = sum(1 for n in nb_tokens if n > limite)
print(f"Morceaux qui dépassent {limite} tokens : {depassements} ({100 * depassements / len(chunks):.1f} %)")

# 3. Afficher les 3 morceaux les plus longs en tokens
pires = sorted(zip(nb_tokens, chunks), key=lambda x: x[0], reverse=True)[:3]
for n, c in pires:
    print(f"\n{n} tokens | {c['source']} page {c['page']}")
    print(c["texte"][:300], "...")