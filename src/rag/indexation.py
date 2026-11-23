"""Module 3, étape B : calcul des plongements et stockage dans ChromaDB."""
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

MODELE = "sentence-transformers/all-MiniLM-L6-v2"
FICHIER_CHUNKS = Path("data/processed/chunks.jsonl")
DOSSIER_BASE = "chroma_db"
NOM_COLLECTION = "articles_falsification"
TAILLE_LOT = 500

# 1. Charger les morceaux
chunks = [json.loads(ligne) for ligne in FICHIER_CHUNKS.open(encoding="utf-8")]
print(f"{len(chunks)} morceaux chargés")

# 2. Calculer les plongements
modele = SentenceTransformer(MODELE)
vecteurs = modele.encode(
    [c["texte"] for c in chunks],
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
)
print(f"Forme du tableau de vecteurs : {vecteurs.shape}")
print(f"Début du premier vecteur : {vecteurs[0][:8]}")

# 3. Recréer la collection à chaque lancement
client = chromadb.PersistentClient(path=DOSSIER_BASE)
try:
    client.delete_collection(NOM_COLLECTION)
except Exception:
    pass  # la collection n'existait pas encore
collection = client.create_collection(
    name=NOM_COLLECTION,
    metadata={"hnsw:space": "cosine"},
)

# 4. Ajouter les morceaux par lots
for i in range(0, len(chunks), TAILLE_LOT):
    lot = chunks[i:i + TAILLE_LOT]
    collection.add(
        ids=[c["id"] for c in lot],
        embeddings=vecteurs[i:i + TAILLE_LOT].tolist(),
        documents=[c["texte"] for c in lot],
        metadatas=[{"source": c["source"], "page": c["page"], "num_chunk": c["num_chunk"]} for c in lot],
    )
print(f"Collection « {NOM_COLLECTION} » : {collection.count()} morceaux indexés dans {DOSSIER_BASE}/")