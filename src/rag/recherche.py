"""Module 3, étape C : rechercher les morceaux les plus proches d'une question."""
import sys

import chromadb
from sentence_transformers import SentenceTransformer

MODELE = "sentence-transformers/all-MiniLM-L6-v2"
DOSSIER_BASE = "chroma_db"
NOM_COLLECTION = "articles_falsification"
K = 5

question = sys.argv[1] if len(sys.argv) > 1 else "How does TruFor detect image forgeries?"

modele = SentenceTransformer(MODELE)
collection = chromadb.PersistentClient(path=DOSSIER_BASE).get_collection(NOM_COLLECTION)

vecteur_question = modele.encode(question, normalize_embeddings=True).tolist()
resultats = collection.query(query_embeddings=[vecteur_question], n_results=K)

print(f"Question : {question}\n")
for rang in range(K):
    distance = resultats["distances"][0][rang]
    meta = resultats["metadatas"][0][rang]
    texte = resultats["documents"][0][rang]
    print(f"{rang + 1}. similarité {1 - distance:.3f} | {meta['source']} page {meta['page']}")
    print(f"   {texte[:250]}...\n")