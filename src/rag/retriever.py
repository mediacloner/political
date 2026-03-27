"""
RAG retriever for long debates (10+ rounds).
Uses ChromaDB + all-MiniLM-L6-v2 (CPU) to store and retrieve prior turns.
"""

import uuid
from typing import Optional


class RAGRetriever:
    def __init__(self, collection_name: str = "debate_turns"):
        self._client = None
        self._collection = None
        self._embed_model = None
        self.collection_name = collection_name
        self._enabled = False

    def _init(self) -> bool:
        """Lazy init — only load when actually needed."""
        if self._enabled:
            return True
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(self.collection_name)
            self._enabled = True
            print("  [rag] ChromaDB + all-MiniLM-L6-v2 initialized", flush=True)
            return True
        except ImportError as e:
            print(f"  [rag] not available: {e}")
            return False

    def add_turn(self, agent: str, round_num: int, content: str) -> None:
        if not self._init():
            return
        doc_id = str(uuid.uuid4())
        self._collection.add(
            documents=[content],
            metadatas=[{"agent": agent, "round": round_num}],
            ids=[doc_id],
        )

    def retrieve(self, query: str, top_k: int = 3, exclude_agent: str = None) -> list[dict]:
        """
        Returns top-K most relevant prior turns for the given query.
        Returns empty list if RAG is not initialized.
        """
        if not self._init() or self._collection.count() == 0:
            return []

        where = {"agent": {"$ne": exclude_agent}} if exclude_agent else None
        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
            where=where,
        )
        output = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            output.append({"content": doc, "agent": meta["agent"], "round": meta["round"]})
        return output

    def format_for_prompt(self, results: list[dict]) -> str:
        if not results:
            return ""
        lines = ["=== RELEVANT PRIOR ARGUMENTS (via RAG) ==="]
        for r in results:
            lines.append(f"\n[{r['agent'].upper()} — Round {r['round']}]\n{r['content']}")
        return "\n".join(lines)
