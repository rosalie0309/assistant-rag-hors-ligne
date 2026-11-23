"""Module 2 : lecture des PDF et découpage en morceaux (chunks).

Les morceaux visent TAILLE_CHUNK caractères, sans jamais dépasser
MAX_TOKENS tokens, la limite du modèle de plongements.
"""
import json
import re
from pathlib import Path

from pypdf import PdfReader
from transformers import AutoTokenizer

DOSSIER_PDF = Path("data/raw")
FICHIER_SORTIE = Path("data/processed/chunks.jsonl")
TAILLE_CHUNK = 800   # cible, en caractères
CHEVAUCHEMENT = 150  # en caractères
MAX_TOKENS = 250     # garantie : limite du modèle (256) moins les tokens [CLS] et [SEP]

TOKENISEUR = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def compter_tokens(texte: str) -> int:
    return len(TOKENISEUR.encode(texte))


def nettoyer(texte: str) -> str:
    texte = re.sub(r"-\n(\w)", r"\1", texte)  # recolle les mots coupés en fin de ligne
    texte = re.sub(r"\s+", " ", texte)         # supprime les espaces et retours à la ligne en trop
    return texte.strip()


def ajuster_fin(texte: str, debut: int, fin: int) -> int:
    """Recule la fin du morceau tant qu'il dépasse MAX_TOKENS."""
    while compter_tokens(texte[debut:fin]) > MAX_TOKENS:
        nouvelle_fin = texte.rfind(". ", debut, fin - 1)  # fin de la phrase précédente
        if nouvelle_fin <= debut + 100:
            nouvelle_fin = texte.rfind(" ", debut, fin - 1)  # sinon, le mot précédent
        if nouvelle_fin <= debut:
            break
        fin = nouvelle_fin + 1
    return fin


def decouper(texte: str, taille: int, chevauchement: int) -> list[str]:
    morceaux = []
    debut = 0
    while debut < len(texte):
        fin = min(debut + taille, len(texte))
        if fin < len(texte):
            # on recule jusqu'à la fin d'une phrase pour ne pas couper au milieu
            point = texte.rfind(". ", debut, fin)
            if point > debut + taille // 2:
                fin = point + 1
        fin = ajuster_fin(texte, debut, fin)  # garantie sur le nombre de tokens
        morceaux.append(texte[debut:fin].strip())
        if fin >= len(texte):
            break
        # le morceau suivant commence dans la zone de chevauchement,
        # de préférence au début d'une phrase, sinon au début d'un mot
        zone = max(fin - chevauchement, debut + 1)
        point = texte.find(". ", zone, fin)
        if point != -1:
            debut = point + 2
        else:
            espace = texte.find(" ", zone, fin)
            debut = espace + 1 if espace != -1 else zone
    return morceaux


def main():
    FICHIER_SORTIE.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with FICHIER_SORTIE.open("w", encoding="utf-8") as sortie:
        for chemin in sorted(DOSSIER_PDF.glob("*.pdf")):
            lecteur = PdfReader(chemin)
            nb_morceaux = 0
            for num_page, page in enumerate(lecteur.pages, start=1):
                texte = nettoyer(page.extract_text() or "")
                if len(texte) < 50:  # page vide ou composée uniquement d'images
                    continue
                for i, morceau in enumerate(decouper(texte, TAILLE_CHUNK, CHEVAUCHEMENT)):
                    chunk = {
                        "id": f"{chemin.stem}_p{num_page}_c{i}",
                        "source": chemin.name,
                        "page": num_page,
                        "num_chunk": i,
                        "texte": morceau,
                    }
                    sortie.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    nb_morceaux += 1
            print(f"{chemin.name} : {len(lecteur.pages)} pages, {nb_morceaux} morceaux")
            total += nb_morceaux
    print(f"Total : {total} morceaux enregistrés dans {FICHIER_SORTIE}")


if __name__ == "__main__":
    main()