from __future__ import annotations

from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        """Answer a question using retrieved chunks as context."""
        if self.store.get_collection_size() == 0:
            return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức (kho dữ liệu rỗng)."

        chunks = self.store.search(question, top_k=top_k)
        if not chunks:
            return "Không tìm thấy đoạn thông tin nào liên quan đến câu hỏi."

        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.get("metadata", {}).get("doc_id", chunk.get("id", "unknown"))
            context_parts.append(f"[{i}] (Nguồn: {source}):\n{chunk['content']}")

        context_text = "\n\n".join(context_parts)
        prompt = (
            "Dưới đây là các đoạn thông tin liên quan được truy xuất từ tài liệu chính sách:\n\n"
            f"{context_text}\n\n"
            f"Dựa duy nhất vào các thông tin trên, hãy trả lời câu hỏi sau (nếu không có thông tin, hãy nói rõ không tìm thấy):\n"
            f"Câu hỏi: {question}\n\n"
            "Câu trả lời:"
        )

        return self.llm_fn(prompt)
