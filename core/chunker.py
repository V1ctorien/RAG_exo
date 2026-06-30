"""
================================================================================
 core/chunker.py — Version simplifiée
================================================================================
 Chunk size fixe : 512 tokens, overlap 15%.
 Pas d'options — juste découpe sémantique propre.
================================================================================
"""

import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("chunker")

MAX_TOKENS    = 512
OVERLAP_RATIO = 0.15
MIN_CHARS     = 40


@dataclass
class Chunk:
    id:          str
    text:        str
    chunk_type:  str
    page:        int
    heading:     str  = ""
    doc_title:   str  = ""
    source_file: str  = ""
    token_count: int  = 0
    img_path:    Optional[str]  = None
    headers:     Optional[list] = None

    def to_chroma_meta(self) -> dict:
        import json
        return {
            "chunk_type":  self.chunk_type,
            "page":        self.page,
            "heading":     self.heading,
            "doc_title":   self.doc_title,
            "source_file": self.source_file,
            "token_count": self.token_count,
            "img_path":    self.img_path or "",
            "headers":     json.dumps(self.headers or [], ensure_ascii=False),
        }


class SemanticChunker:
    """Découpe sémantique avec paramètres fixes."""

    @staticmethod
    def _tok(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def _sentences(text: str) -> list[str]:
        text = re.sub(r'\n{3,}', '\n\n', text.strip())
        parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-Ü\d«"])', text)
        return [p.strip() for p in parts if p.strip()]

    def _make(self, text, chunk_type, page, heading, doc_title, source_file,
              img_path=None, headers=None) -> Chunk:
        return Chunk(
            id=str(uuid.uuid4()),
            text=text.strip(),
            chunk_type=chunk_type,
            page=page,
            heading=heading,
            doc_title=doc_title,
            source_file=source_file,
            token_count=self._tok(text),
            img_path=img_path,
            headers=headers,
        )

    def _sliding_window(self, sentences, heading, page, doc_meta) -> list[Chunk]:
        chunks, current, cur_tok, prev_overlap = [], [], 0, ""
        overlap_max = int(MAX_TOKENS * OVERLAP_RATIO)

        for s in sentences:
            st = self._tok(s)
            if cur_tok + st > MAX_TOKENS and current:
                text = " ".join(current)
                chunks.append(self._make(text, "section", page, heading, **doc_meta))

                olap, olap_tok = [], 0
                for x in reversed(current):
                    t = self._tok(x)
                    if olap_tok + t > overlap_max:
                        break
                    olap.insert(0, x)
                    olap_tok += t

                current, cur_tok = olap[:], olap_tok

            current.append(s)
            cur_tok += st

        if current:
            text = " ".join(current)
            if len(text) >= MIN_CHARS:
                chunks.append(self._make(text, "section", page, heading, **doc_meta))

        return chunks

    def chunk(self, extracted: dict) -> list[Chunk]:
        log.info("Chunking…")
        meta = extracted.get("metadata", {})
        doc_meta = {
            "doc_title":   meta.get("title") or Path(extracted.get("source_file", "doc")).stem,
            "source_file": extracted.get("source_file", ""),
        }

        all_chunks: list[Chunk] = []
        current_heading = ""
        text_buffer: list[str] = []

        def flush():
            nonlocal text_buffer
            if not text_buffer:
                return
            full = "\n".join(text_buffer)
            if len(full) >= MIN_CHARS:
                all_chunks.extend(
                    self._sliding_window(self._sentences(full),
                                         current_heading, page_num, doc_meta)
                )
            text_buffer.clear()

        for page_data in extracted.get("pages", []):
            page_num = page_data["page"]

            for block in page_data.get("blocks", []):
                btype = block.get("type")

                if btype == "text":
                    text = block.get("text", "").strip()
                    if not text:
                        continue
                    if block.get("role") == "heading":
                        flush()
                        current_heading = text
                        text_buffer.append(text)
                    else:
                        text_buffer.append(text)

                elif btype == "table":
                    flush()
                    md = block.get("markdown", "")
                    if md and len(md) >= MIN_CHARS:
                        all_chunks.append(self._make(
                            md, "table", page_num, current_heading,
                            headers=block.get("headers"), **doc_meta,
                        ))

                elif btype == "image":
                    ocr = block.get("ocr_text", "").strip()
                    if ocr and len(ocr) >= MIN_CHARS:
                        flush()
                        all_chunks.append(self._make(
                            f"[Figure p.{page_num}] {ocr}",
                            "figure", page_num, current_heading,
                            img_path=block.get("img_path"), **doc_meta,
                        ))

            flush()

        log.info(f"  → {len(all_chunks)} chunks")
        return all_chunks
