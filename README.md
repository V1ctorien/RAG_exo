# 📚 RAG — Chat with PDF Documents
### Projet Final — Présentation 30 Juin 2026

Using Python 3.10.20 environment at: ninenv

> Système de Retrieval-Augmented Generation (RAG) permettant de **poser des
> questions en langage naturel sur vos documents PDF**, avec réponses sourcées
> et interface web intuitive.

---

## 🏗️ Architecture complète

```
                    ┌─────────────────────────────────────────────────┐
                    │                   PDF d'entrée                  │
                    └───────────────────┬─────────────────────────────┘
                                        │
                    ┌───────────────────▼─────────────────────────────┐
                    │  COUCHE 1 — core/pdf_extractor.py               │
                    │                                                  │
                    │  PyMuPDF  → texte natif, hiérarchie, images     │
                    │  pdfplumber → tableaux vectoriels (cellules)     │
                    └───────────────────┬─────────────────────────────┘
                                        │  JSON structuré
                    ┌───────────────────▼─────────────────────────────┐
                    │  COUCHE 2 — core/chunker.py                     │
                    │                                                  │
                    │  Sliding window (512 tokens, 15% overlap)       │
                    │  Frontières sémantiques (headings)              │
                    │  Chunks atomiques (tables, figures, formules)   │
                    └───────────────────┬─────────────────────────────┘
                                        │  List[Chunk]
                    ┌───────────────────▼─────────────────────────────┐
                    │  COUCHE 3 — core/vector_store.py                │
                    │                                                  │
                    │  all-mpnet-base-v2 (HuggingFace, GPU FP16)     │
                    │  ChromaDB — index HNSW cosinus, persistant      │
                    └───────────────────┬─────────────────────────────┘
                                        │  Vecteurs indexés
                    ┌───────────────────▼─────────────────────────────┐
                    │  COUCHE 4 — core/rag_pipeline.py                │
                    │                                                  │
                    │  Retrieval ChromaDB (top-k chunks)              │
                    │  LangChain LCEL (prompt | llm | parser)         │
                    │  LLM HuggingFace :                              │
                    │    • google/flan-t5-large       (léger/CPU)     │
                    │    • Mistral-7B-Instruct-v0.1   (GPU 4-bit)     │
                    └───────────────────┬─────────────────────────────┘
                                        │
                    ┌───────────────────▼─────────────────────────────┐
                    │  UI — ui/gradio_app.py                          │
                    │                                                  │
                    │  Upload PDF  → Indexation avec progress bar     │
                    │  Chat        → Streaming token par token        │
                    │  Sources     → Affichage page + score + extrait │
                    │  Résumé      → Généré automatiquement           │
                    └─────────────────────────────────────────────────┘
```

---

## 📁 Structure des fichiers

```
rag_project/
├── main.py                  # Point d'entrée (CLI + UI)
├── requirements.txt         # Dépendances Python
├── README.md                # Ce fichier
│
├── core/
│   ├── pdf_extractor.py     # Couche 1 : extraction PDF
│   ├── chunker.py           # Couche 2 : chunking sémantique
│   ├── vector_store.py      # Couche 3 : embeddings + ChromaDB
│   └── rag_pipeline.py      # Couche 4 : RAG LangChain + LLM
│
└── ui/
    └── gradio_app.py        # Interface web Gradio
```

---

## 🚀 Installation

### Prérequis
- Python 3.10+
- GPU recommandé (8 Go VRAM) — flan-t5 fonctionne aussi sur CPU

### Étapes

```bash
# 1. Cloner / copier le projet
git clone <votre-repo>
cd rag_project

# 2. Environnement virtuel
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

# 3. PyTorch GPU (adapter selon votre CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Toutes les dépendances
pip install -r requirements.txt
```

---

## 💻 Utilisation

### Interface web (recommandée pour la démo)
```bash
python main.py --ui
# → Ouvrir http://localhost:7860
```

### CLI — Question rapide
```bash
python main.py --pdf rapport.pdf --query "Quelles sont les conclusions ?"
```

### CLI — Mode interactif
```bash
python main.py --pdf rapport.pdf
# → Chat en boucle dans le terminal
```

### CLI — Résumé automatique
```bash
python main.py --pdf rapport.pdf --summarize
```

### Avec Mistral-7B (GPU, meilleure qualité)
```bash
python main.py --ui \
    --llm-model mistralai/Mistral-7B-Instruct-v0.1 \
    --use-4bit        # quantification 4-bit → ~2.5 Go VRAM
```

---

## 🎛️ Options principales

| Option | Défaut | Description |
|--------|--------|-------------|
| `--ui` | — | Lance l'interface Gradio |
| `--pdf` | — | Chemin du fichier PDF |
| `--query` | — | Question à poser |
| `--llm-model` | `google/flan-t5-large` | Modèle de génération |
| `--embed-model` | `all-mpnet-base-v2` | Modèle d'embedding |
| `--chunk-size` | `512` | Tokens max par chunk |
| `--top-k` | `5` | Chunks récupérés par requête |
| `--use-4bit` | `False` | Quantification 4-bit (Mistral) |
| `--ocr` | `False` | OCR sur les images (EasyOCR) |
| `--reindex` | `False` | Vider et re-indexer |

---

## 📊 Budget VRAM (GPU 8 Go)

| Composant | Modèle | VRAM |
|-----------|--------|------|
| Embeddings | all-mpnet-base-v2 FP16 | ~200 Mo |
| LLM option A | flan-t5-large | ~1.5 Go |
| LLM option B | Mistral-7B 4-bit | ~2.5 Go |
| **Total A (recommandé démo)** | | **~1.8 Go** |
| **Total B (meilleure qualité)** | | **~2.8 Go** |

---

## 🔬 Explication des choix techniques

### Pourquoi all-mpnet-base-v2 ?
Recommandé explicitement dans le cahier des charges. Score MTEB élevé,
multilingue partiel, dimension 768, bien adapté aux documents techniques.

### Pourquoi flan-t5-large pour la démo ?
- Léger (~3 Go), fonctionne sur CPU si besoin
- Seq2seq = génère directement la réponse sans répéter le prompt
- Très stable, pas d'hallucinations excessives sur du texte factuel

### Pourquoi Mistral-7B pour la qualité ?
- Meilleur modèle instruction-following open source en 7B
- 4-bit quantization = ~2.5 Go VRAM seulement
- Réponses plus naturelles et plus longues

### Pourquoi ChromaDB ?
- Zéro configuration serveur (local, persistant)
- Index HNSW = O(log n) en recherche → très rapide même sur 100k chunks
- Filtres sur métadonnées (page, type) intégrés

### Chunking sémantique vs fixe ?
Le chunking fixe (ex: 512 tokens à la suite) coupe les phrases au milieu.
Notre approche respecte :
1. Les frontières de section (headings)
2. Les fins de phrases
3. L'overlap 15% pour ne pas perdre le contexte aux bords

---

## 📹 Démo vidéo (pour le rapport)

Points à montrer :
1. Import d'un PDF et barre de progression
2. Résumé automatique généré
3. Question simple avec citation [p.X]
4. Filtrage par type (tableau uniquement)
5. Comparaison flan-t5 vs Mistral-7B sur la même question
