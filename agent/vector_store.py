from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

load_dotenv()


class VectorStore:
    def __init__(
        self,
        data_dir: str = "data",
        persist_directory: Optional[str] = None,
        collection_name: str = "agent_docs",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        self.data_dir = Path(data_dir)
        self.persist_directory = Path(persist_directory) if persist_directory else Path(__file__).parent / "chroma_db"
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory),
        )

    def _resolve_data_files(self, data_files: Optional[List[str]] = None) -> List[Path]:
        if data_files:
            return [Path(file_path) for file_path in data_files]

        ignored_files = {"HARD_CASES_GUIDE.md", "golden_set.jsonl"}
        return sorted(
            file_path
            for file_path in self.data_dir.glob("*")
            if file_path.is_file()
            and file_path.suffix.lower() in {".md", ".txt"}
            and file_path.name not in ignored_files
        )

    def _chunk_markdown_document(self, file_path: Path) -> List[Document]:
        doc_id = file_path.stem

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        raw_text = file_path.read_text(encoding="utf-8")
        sections = header_splitter.split_text(raw_text)

        chunks: List[Document] = []
        chunk_index = 1

        for section in sections:
            section_chunks = text_splitter.split_documents([section])

            for chunk in section_chunks:
                chunk_id = f"{doc_id}_chunk_{chunk_index:03d}"
                chunk.metadata.update(
                    {
                        "source_doc_id": doc_id,
                        "source_chunk_id": chunk_id,
                        "source_file": str(file_path),
                    }
                )
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def load_documents(self, data_files: Optional[List[str]] = None) -> List[Document]:
        documents: List[Document] = []

        for file_path in self._resolve_data_files(data_files):
            documents.extend(self._chunk_markdown_document(file_path))

        return documents

    def upsert(self, data_files: Optional[List[str]] = None) -> int:
        """
        Embed and upload chunks to the local Chroma collection.

        Returns the number of chunks uploaded.
        """
        documents = self.load_documents(data_files=data_files)
        if not documents:
            return 0

        ids = [doc.metadata["source_chunk_id"] for doc in documents]

        # Chroma add_documents may reject duplicate IDs, so delete first to make
        # repeated upserts deterministic.
        try:
            self.store.delete(ids=ids)
        except Exception:
            pass

        self.store.add_documents(documents=documents, ids=ids)

        persist = getattr(self.store, "persist", None)
        if callable(persist):
            persist()

        return len(documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search the Chroma collection and return scored chunks.
        """
        results = self.store.similarity_search_with_score(query, k=top_k)

        return [
            {
                "id": doc.metadata.get("source_chunk_id"),
                "content": doc.page_content,
                "score": score,
                "metadata": doc.metadata,
            }
            for doc, score in results
        ]


if __name__ == "__main__":
    vector_store = VectorStore()
    count = vector_store.upsert()
    print(f"Uploaded {count} chunks to {vector_store.persist_directory}")

    for item in vector_store.search("Sinh vien duoc muon toi da bao nhieu cuon sach?", top_k=3):
        print(item["id"], item["score"])
