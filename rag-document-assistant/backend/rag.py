"""
Minimal RAG (Retrieval-Augmented Generation) engine.

Steps
1. Load  : read .txt/.md files from /docs
2. Chunk : split each document into overlapping chunks
3. Embed : turn chunks into vectors (TF-IDF by default; optional sentence-transformers)
4. Index : keep vectors in a small in-memory vector store
5. Retrieve: cosine similarity search for the top-k chunks of a question
6. Generate: send question + retrieved chunks to an LLM (optional; falls back to
             returning the best matching passages, so it works with NO API key)
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DOCS_DIR = Path(__file__).parent / "docs"


@dataclass
class Chunk:
    text: str
    source: str


def load_documents(folder: Path = DOCS_DIR) -> list[tuple[str, str]]:
    return [(p.name, p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*")) if p.suffix in {".txt", ".md"}]


def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Split on markdown headings first (keeps each topic together), then by words
    with overlap so long sections are not cut mid-idea."""
    sections = [s.strip() for s in re.split(r"\n(?=#{1,3} )", text) if s.strip()]
    chunks = []
    step = size - overlap
    for section in sections:
        words = section.split()
        if len(words) < 8:   # skip heading-only fragments such as a bare document title
            continue
        chunks += [" ".join(words[i:i + size]) for i in range(0, len(words), step)]
    return chunks


STOP = {"a", "an", "the", "is", "are", "do", "does", "i", "my", "me", "to", "of", "in", "for", "and",
        "or", "on", "what", "how", "can", "get", "it", "be", "must", "per", "at", "by", "with", "from"}


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, drop stop words, and strip simple plurals
    (so 'passwords' matches 'password')."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in tokens if t not in STOP]


class VectorStore:
    """Tiny vector database: stores a matrix of vectors + metadata, searches by cosine similarity."""

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.vectors = None
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                          ngram_range=(1, 2))

    def build(self, docs: list[tuple[str, str]]) -> None:
        self.chunks = [Chunk(t, name) for name, text in docs for t in chunk_text(text)]
        # TF-IDF rows are L2-normalised, so a dot product == cosine similarity
        self.vectors = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = self.vectorizer.transform([query])
        scores = (self.vectors @ q.T).toarray().ravel()
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]


def generate_answer(question: str, hits: list[tuple[Chunk, float]]) -> str:
    """Use an LLM if a key is configured, otherwise return the retrieved passages."""
    if not hits:
        return "I could not find anything relevant in the documents."

    context = "\n\n".join(f"[{c.source}] {c.text}" for c, _ in hits)
    prompt = (f"Answer the question using ONLY the context below. If the answer is not "
              f"in the context, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}")

    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(model=os.getenv("LLM_MODEL", "claude-sonnet-5-5"),
                                     max_tokens=500, messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                                              messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content

    # No LLM configured: return best passage (retrieval-only mode)
    return "(Retrieval-only mode - set ANTHROPIC_API_KEY or OPENAI_API_KEY for generated answers)\n\n" + hits[0][0].text


store = VectorStore()
store.build(load_documents())


def ask(question: str, k: int = 3) -> dict:
    hits = store.search(question, k)
    return {
        "answer": generate_answer(question, hits),
        "sources": [{"source": c.source, "score": round(s, 3), "text": c.text[:300]} for c, s in hits],
    }
