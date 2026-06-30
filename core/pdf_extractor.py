"""
================================================================================
 core/pdf_extractor.py
 COUCHE 1 — Extraction PDF avec PyMuPDF + pdfplumber
================================================================================
 Rôle : transformer un PDF brut en JSON structuré (texte, tableaux, images).
 C'est le premier maillon du pipeline RAG.

 Basé sur pdf_pipeline.py (session précédente), version allégée :
   → PyMuPDF    : texte natif + hiérarchie (taille fonte) + images vers disk
   → pdfplumber : tableaux vectoriels cellule par cellule
================================================================================
"""

import logging
import re
from pathlib import Path

log = logging.getLogger("pdf_extractor")


def extract_pdf(pdf_path: str, img_dir: str = None, use_ocr: bool = False) -> dict:
    """
    Extrait tout le contenu d'un PDF et le retourne en dict structuré.

    Args:
        pdf_path : chemin vers le .pdf
        img_dir  : dossier pour sauvegarder les images extraites
        use_ocr  : lancer EasyOCR sur les figures (False = plus rapide)

    Returns:
        {
          "metadata":    { title, author, … },
          "n_pages":     int,
          "source_file": str,
          "pages":       [ { "page": int, "blocks": [...] } ]
        }
    """
    import fitz        # pip install pymupdf
    import pdfplumber  # pip install pdfplumber

    pdf_path = str(pdf_path)
    log.info(f"Extraction PDF : {Path(pdf_path).name}")

    if img_dir:
        Path(img_dir).mkdir(parents=True, exist_ok=True)

    # ── Tableaux via pdfplumber (en premier pour éviter double ouverture) ─
    tables_by_page = _extract_tables(pdf_path)

    # ── Ouverture PyMuPDF ─────────────────────────────────────────────────
    doc = fitz.open(pdf_path)

    result = {
        "metadata":    dict(doc.metadata),
        "n_pages":     len(doc),
        "source_file": pdf_path,
        "pages":       [],
    }

    for page_num, page in enumerate(doc, start=1):
        text_dict  = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES)
        blocks_out = []

        for block in text_dict["blocks"]:

            # ── Bloc texte natif ──────────────────────────────────────────
            if block["type"] == 0:
                spans, sizes = [], []
                for line in block["lines"]:
                    for span in line["spans"]:
                        t = span["text"].strip()
                        if t:
                            spans.append(t)
                            sizes.append(span["size"])

                if not spans:
                    continue

                full_text = " ".join(spans)
                avg_size  = sum(sizes) / len(sizes)

                blocks_out.append({
                    "type":      "text",
                    "role":      _detect_role(full_text, avg_size),
                    "text":      full_text,
                    "font_size": round(avg_size, 1),
                    "bbox":      block["bbox"],
                })

            # ── Image embarquée ───────────────────────────────────────────
            elif block["type"] == 1 and img_dir:
                xref = block.get("image")
                if xref:
                    try:
                        img_data = doc.extract_image(xref)
                        img_path = Path(img_dir) / f"p{page_num}_img{xref}.{img_data['ext']}"
                        img_path.write_bytes(img_data["image"])
                        blocks_out.append({
                            "type":     "image",
                            "role":     "figure",
                            "bbox":     block["bbox"],
                            "img_path": str(img_path),
                            "ocr_text": "",
                        })
                    except Exception as e:
                        log.warning(f"Image ignorée (xref={xref}): {e}")

        # ── Tableaux pdfplumber pour cette page ───────────────────────────
        for raw_table in tables_by_page.get(page_num, []):
            if not raw_table:
                continue
            headers = [c or "" for c in raw_table[0]]
            rows    = [[c or "" for c in row] for row in raw_table[1:]]
            blocks_out.append({
                "type":     "table",
                "headers":  headers,
                "rows":     rows,
                "markdown": _table_to_markdown(headers, rows),
            })

        result["pages"].append({
            "page":   page_num,
            "width":  page.rect.width,
            "height": page.rect.height,
            "blocks": blocks_out,
        })

    doc.close()

    if use_ocr:
        _run_ocr(result)

    n_blocks = sum(len(p["blocks"]) for p in result["pages"])
    log.info(f"  → {result['n_pages']} pages | {n_blocks} blocs extraits")
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_tables(pdf_path: str) -> dict:
    import pdfplumber
    tables = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                t = page.extract_tables()
                if t:
                    tables[i] = t
    except Exception as e:
        log.warning(f"pdfplumber: {e}")
    return tables


def _detect_role(text: str, font_size: float) -> str:
    """Heuristique : détecte heading / caption / body selon fonte + contenu."""
    low = text.lower().strip()
    if re.match(r"^(figure|fig\.|tableau|table|annexe)\s", low):
        return "caption"
    if font_size >= 13 or (font_size >= 11 and len(text) < 80 and text[0].isupper()):
        return "heading"
    return "body"


def _table_to_markdown(headers: list, rows: list) -> str:
    if not headers:
        return ""
    h   = "| " + " | ".join(str(c) for c in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    bdy = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join(filter(None, [h, sep, bdy]))


def _run_ocr(result: dict) -> None:
    """EasyOCR léger sur les figures extraites (optionnel)."""
    try:
        import easyocr
        reader = easyocr.Reader(["fr", "en"], gpu=True)
        for page in result["pages"]:
            for block in page["blocks"]:
                if block["type"] == "image" and block.get("img_path"):
                    try:
                        block["ocr_text"] = " ".join(
                            reader.readtext(block["img_path"], detail=0)
                        )
                    except Exception as e:
                        log.warning(f"OCR image: {e}")
    except ImportError:
        log.warning("easyocr non installé — images sans OCR")
