"""
================================================================================
 core/rag_pipeline.py — Version simplifiée
================================================================================
 Modèle fixe : google/flan-t5-large (CPU, léger, 3 Go)
 Pas de choix, pas d'options — juste retrieval + génération.
================================================================================
"""

import logging
from typing import Optional

log = logging.getLogger("rag_pipeline")

SYSTEM_PROMPT = """Tu es un assistant expert en analyse documentaire.
Réponds UNIQUEMENT à partir du contexte fourni.
Pour chaque information, cite la page source entre crochets [p.X].
Si la réponse n'est pas dans le contexte, dis-le clairement.
Réponds en français."""

RAG_TEMPLATE = """CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE (avec citations [p.X]) :"""


class RAGPipeline:
    def __init__(self, vector_store, top_k: int = 5):
        self.store = vector_store
        self.top_k = top_k
        self.llm   = _load_flan_t5()

        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=f"{SYSTEM_PROMPT}\n\n{RAG_TEMPLATE}",
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
        log.info("RAGPipeline prêt.")

    def query(self, question: str) -> dict:
        """Recherche dans tous les chunks et génère une réponse."""
        results = self.store.search(question, k=self.top_k)
        if not results:
            return {"question": question, "answer": "Aucun document pertinent trouvé.", "sources": []}

        context = _format_context(results)
        answer  = self.chain.invoke({"context": context, "question": question})

        return {
            "question": question,
            "answer":   answer.strip(),
            "sources":  _format_sources(results),
        }


def _load_flan_t5():
    """
    Charge flan-t5-large via un wrapper LangChain custom.
    text2text-generation supprimé des nouvelles versions transformers →
    on appelle model.generate() directement.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from langchain_core.language_models.llms import LLM
    from typing import Optional, List

    MODEL_NAME = "google/flan-t5-large"
    log.info(f"Chargement {MODEL_NAME}…")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    device = next(model.parameters()).device
    log.info(f"  flan-t5-large chargé sur {device} ✓")

    class FlanT5(LLM):
        @property
        def _llm_type(self) -> str:
            return "flan-t5"

        def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
            inputs  = tokenizer(prompt, return_tensors="pt",
                                max_length=1024, truncation=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=512,
                                     do_sample=False, num_beams=2)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

    return FlanT5()


def _format_context(results: list) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        parts.append(
            f"[Source {i} — Page {meta.get('page','?')}]\n{r['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _format_sources(results: list) -> list:
    return [
        {
            "text":    r["text"][:250] + ("…" if len(r["text"]) > 250 else ""),
            "page":    r["metadata"].get("page"),
            "heading": r["metadata"].get("heading", ""),
            "type":    r["metadata"].get("chunk_type", ""),
            "score":   r["score"],
        }
        for r in results
    ]
