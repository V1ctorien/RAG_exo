"""
================================================================================
 main.py — Point d'entrée principal
================================================================================
 Deux modes d'utilisation :

   1. Interface web (Gradio) :
      python main.py --ui
      → Ouvre http://localhost:7860

   2. Mode CLI (terminal) :
      python main.py --pdf mon_doc.pdf --query "Ma question ?"
      → Répond dans le terminal, sauvegarde le JSON

 Architecture du projet :
   main.py              ← ce fichier (orchestrateur)
   core/
     pdf_extractor.py   ← Couche 1 : extraction PyMuPDF + pdfplumber
     chunker.py         ← Couche 2 : découpe sémantique
     vector_store.py    ← Couche 3 : embeddings HF + ChromaDB
     rag_pipeline.py    ← Couche 4 : RAG LangChain + LLM HF
   ui/
     gradio_app.py      ← Interface web
================================================================================
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


def run_cli(args):
    """Mode ligne de commande : indexe un PDF et répond à une question."""
    from core.pdf_extractor import extract_pdf
    from core.chunker       import SemanticChunker
    from core.vector_store  import VectorStore
    from core.rag_pipeline  import RAGPipeline

    pdf_path = args.pdf
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Extraction ─────────────────────────────────────────────────────
    log.info(f"=== 1/4 Extraction PDF : {pdf_path} ===")
    extracted = extract_pdf(
        pdf_path,
        img_dir=str(out_dir / "images"),
        use_ocr=args.ocr,
    )
    json_path = out_dir / "extracted.json"
    json_path.write_text(
        json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(f"  → JSON sauvegardé : {json_path}")

    # ── 2. Chunking ───────────────────────────────────────────────────────
    log.info(f"=== 2/4 Chunking (max_tokens={args.chunk_size}) ===")
    chunker = SemanticChunker(max_tokens=args.chunk_size)
    chunks  = chunker.chunk(extracted)
    log.info(f"  → {len(chunks)} chunks générés")

    # ── 3. Embedding + ChromaDB ───────────────────────────────────────────
    log.info(f"=== 3/4 Indexation ChromaDB (modèle={args.embed_model}) ===")
    collection = Path(pdf_path).stem.replace(" ", "_").lower()[:40]
    store = VectorStore(
        collection_name=collection,
        persist_dir=str(out_dir / "chroma_db"),
        model_name=args.embed_model,
    )
    if args.reindex:
        store.clear()
    store.add_chunks(chunks)

    # ── 4. RAG ────────────────────────────────────────────────────────────
    log.info(f"=== 4/4 RAG (LLM={args.llm_model}) ===")
    rag = RAGPipeline(
        vector_store=store,
        llm_model=args.llm_model,
        top_k=args.top_k,
        use_4bit=args.use_4bit,
    )

    if args.query:
        log.info(f"  Question : {args.query}")
        result = rag.query(args.query)

        print("\n" + "═" * 70)
        print(f"  QUESTION : {result['question']}")
        print("═" * 70)
        print(f"  RÉPONSE  :\n\n  {result['answer']}")
        print("\n  SOURCES :")
        for src in result["sources"]:
            print(f"    [p.{src['page']}] ({src['type']}, {src['score']:.0%}) "
                  f"{src['text'][:80]}…")
        print("═" * 70 + "\n")

        # Sauvegarde la réponse
        (out_dir / "answer.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    elif args.summarize:
        summary = rag.summarize()
        print("\n📝 RÉSUMÉ DU DOCUMENT\n" + "─" * 40)
        print(summary)

    else:
        # Mode interactif
        print("\n💬 Mode interactif (Ctrl+C pour quitter)\n")
        while True:
            try:
                question = input("Question > ").strip()
                if not question:
                    continue
                result = rag.query(question)
                print(f"\n🤖 {result['answer']}\n")
                for s in result["sources"][:3]:
                    print(f"   [p.{s['page']}] {s['text'][:80]}…")
                print()
            except KeyboardInterrupt:
                print("\nAu revoir !")
                break


def run_ui():
    """Mode interface web Gradio."""
    from ui.gradio_app import build_ui
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG Pipeline — Chat avec vos PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Interface web
  python main.py --ui

  # CLI : indexer + question
  python main.py --pdf doc.pdf --query "Quel est l'objectif ?"

  # CLI : mode interactif
  python main.py --pdf doc.pdf

  # Avec Mistral-7B en 4-bit
  python main.py --pdf doc.pdf --llm mistralai/Mistral-7B-Instruct-v0.1 --use-4bit --ui
        """,
    )

    parser.add_argument("--ui",          action="store_true",
                        help="Lancer l'interface Gradio")
    parser.add_argument("--pdf",         type=str,
                        help="Chemin vers le fichier PDF")
    parser.add_argument("--query",       type=str,
                        help="Question à poser (mode CLI)")
    parser.add_argument("--summarize",   action="store_true",
                        help="Générer un résumé du document")
    parser.add_argument("--out",         type=str, default="./output",
                        help="Dossier de sortie (défaut: ./output)")
    parser.add_argument("--embed-model", type=str,
                        default="sentence-transformers/all-mpnet-base-v2",
                        help="Modèle d'embedding HuggingFace")
    parser.add_argument("--llm-model",   type=str,
                        default="google/flan-t5-large",
                        help="Modèle LLM HuggingFace")
    parser.add_argument("--chunk-size",  type=int, default=512,
                        help="Taille max des chunks en tokens")
    parser.add_argument("--top-k",       type=int, default=5,
                        help="Nb de chunks récupérés par requête")
    parser.add_argument("--use-4bit",    action="store_true",
                        help="Quantification 4-bit pour le LLM")
    parser.add_argument("--ocr",         action="store_true",
                        help="Activer l'OCR sur les images (EasyOCR)")
    parser.add_argument("--reindex",     action="store_true",
                        help="Vider et re-indexer la collection ChromaDB")

    args = parser.parse_args()

    if args.ui:
        run_ui()
    elif args.pdf:
        run_cli(args)
    else:
        parser.print_help()
