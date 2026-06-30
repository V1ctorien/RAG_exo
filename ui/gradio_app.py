"""
================================================================================
 ui/gradio_app.py — Version simplifiée
================================================================================
 Interface épurée : upload → indexer → question. C'est tout.
================================================================================
"""

import logging
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr

from core.pdf_extractor import extract_pdf
from core.chunker import SemanticChunker
from core.vector_store import VectorStore
from core.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("gradio_app")

app_state = {"rag": None}


def upload_and_index(pdf_file, progress=gr.Progress()):
    if pdf_file is None:
        return "⚠️ Aucun fichier sélectionné.", ""

    pdf_path = pdf_file.name
    doc_name = Path(pdf_path).stem

    try:
        progress(0.1, desc="📄 Lecture du PDF…")
        extracted = extract_pdf(pdf_path, img_dir=tempfile.mkdtemp(), use_ocr=False)
        n_pages   = extracted["n_pages"]
        progress(0.35, desc=f"✅ {n_pages} pages")

        progress(0.4, desc="✂️ Chunking…")
        chunks   = SemanticChunker().chunk(extracted)
        n_chunks = len(chunks)
        progress(0.55, desc=f"✅ {n_chunks} chunks")

        progress(0.6, desc="🔢 Embeddings…")
        collection = re.sub(r"[^a-z0-9_]", "_", doc_name.lower())[:50]
        store = VectorStore(collection_name=collection,
                            persist_dir=f"./chroma_db/{collection}")
        store.clear()
        store.add_chunks(chunks)
        progress(0.8, desc="✅ Index créé")

        progress(0.85, desc="🤖 Chargement LLM…")
        app_state["rag"] = RAGPipeline(vector_store=store)
        progress(1.0, desc="✅ Prêt !")

        meta  = extracted.get("metadata", {})
        title = meta.get("title") or doc_name
        info  = f"**📄 {title}** | 📃 {n_pages} pages | 🧩 {n_chunks} chunks"
        return f"✅ Prêt !", info

    except Exception as e:
        log.error(e, exc_info=True)
        return f"❌ {str(e)}", ""


def chat(message: str, history: list):
    if not message.strip():
        return history, ""

    rag = app_state.get("rag")
    if rag is None:
        return history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": "⚠️ Importez d'abord un PDF."},
        ], ""

    result = rag.query(message)

    sources = []
    icons = {"section": "📝", "table": "📊", "figure": "🖼️"}
    for i, s in enumerate(result["sources"], 1):
        icon = icons.get(s.get("type", ""), "📄")
        hdg  = f" *{s['heading']}*" if s.get("heading") else ""
        txt  = s["text"][:200] + ("…" if len(s["text"]) > 200 else "")
        sources.append(f"**{icon} [{i}] p.{s.get('page','?')}**{hdg} `{s['score']:.0%}`\n> {txt}")

    sources_md = "### 📚 Sources\n\n" + "\n\n".join(sources) if sources else ""

    return history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": result["answer"]},
    ], sources_md


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="📚 Chat PDF") as demo:

        gr.HTML("""
        <div style="text-align:center;padding:20px 0 8px">
          <h1 style="font-size:1.8rem;font-weight:700">📚 Chat avec votre PDF</h1>
          <p style="color:#888;font-size:.9rem">
            Importez un PDF et posez vos questions.
          </p>
        </div>
        """)

        with gr.Row(equal_height=False):

            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### 📄 Document")
                pdf_upload   = gr.File(label="Votre PDF", file_types=[".pdf"])
                index_btn    = gr.Button("🚀 Indexer", variant="primary", size="lg")
                status_box   = gr.Markdown("*En attente…*")
                doc_info_box = gr.Markdown("")

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=460,
                    placeholder="*La réponse apparaîtra ici…*",
                    render_markdown=True,
                )
                with gr.Row():
                    msg_tb   = gr.Textbox(placeholder="Votre question…", label="", scale=5)
                    send_btn = gr.Button("Envoyer ▶", variant="primary", scale=1)
                clear_btn   = gr.Button("🗑️ Effacer", variant="secondary", size="sm")
                sources_box = gr.Markdown("*Les sources apparaîtront ici.*")

        gr.Examples(
            examples=[
                ["Quel est l'objectif de ce document ?"],
                ["Quelles sont les conclusions ?"],
                ["Résume les points clés."],
                ["Quelles statistiques sont mentionnées ?"],
            ],
            inputs=[msg_tb],
        )

        index_btn.click(upload_and_index, [pdf_upload], [status_box, doc_info_box])
        send_btn.click(chat, [msg_tb, chatbot], [chatbot, sources_box]).then(lambda: "", outputs=[msg_tb])
        msg_tb.submit(chat, [msg_tb, chatbot], [chatbot, sources_box]).then(lambda: "", outputs=[msg_tb])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, sources_box])

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
