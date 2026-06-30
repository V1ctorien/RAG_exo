"""
================================================================================
 core/vector_store.py — Version simplifiée
================================================================================
 Modèle fixe : sentence-transformers/all-MiniLM-L6-v2 (90 Mo, rapide)
 ChromaDB local persistant.
================================================================================
"""

import logging
from typing import Optional

import torch

log = logging.getLogger("vector_store")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStore:
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_dir:     str = "./chroma_db",
    ):
        from sentence_transformers import SentenceTransformer
        import chromadb
        from chromadb.config import Settings

        self.collection_name = collection_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        log.info(f"VectorStore | {EMBED_MODEL} | {self.device}")

        # Modèle d'embedding fixe
        self.model = SentenceTransformer(EMBED_MODEL, device=self.device)
        if self.device == "cuda":
            self.model = self.model.half()

        # ChromaDB persistant
        self._chroma = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(f"  Collection '{collection_name}' — {self._col.count()} docs")

    def add_chunks(self, chunks) -> None:
        """Encode et indexe les chunks."""
        if not chunks:
            return

        log.info(f"Indexation de {len(chunks)} chunks…")

        texts = [
            f"Contexte : {c.heading}\n{c.text}" if c.heading and c.heading not in c.text
            else c.text
            for c in chunks
        ]

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Upsert par batch
        BATCH = 500
        for i in range(0, len(chunks), BATCH):
            bc = chunks[i:i + BATCH]
            be = embeddings[i:i + BATCH]
            self._col.upsert(
                ids=[c.id for c in bc],
                embeddings=be.tolist(),
                documents=[c.text for c in bc],
                metadatas=[c.to_chroma_meta() for c in bc],
            )

        log.info(f"  → {self._col.count()} documents en base")

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Recherche les k chunks les plus proches."""
        q_vec = self.model.encode(
            f"query: {query}",
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        n = min(k, self._col.count())
        if n == 0:
            return []

        raw = self._col.query(
            query_embeddings=[q_vec],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        return [
            {
                "text":     doc,
                "score":    round(1.0 - dist, 4),
                "metadata": meta,
            }
            for doc, meta, dist in zip(
                raw["documents"][0],
                raw["metadatas"][0],
                raw["distances"][0],
            )
        ]

    def clear(self) -> None:
        """Vide la collection."""
        self._chroma.delete_collection(self.collection_name)
        self._col = self._chroma.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(f"Collection '{self.collection_name}' vidée.")

    @property
    def count(self) -> int:
        return self._col.count()
