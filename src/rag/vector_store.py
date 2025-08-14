from typing import List, Dict, Any, Optional
from chromadb import Client
from chromadb.config import Settings
from pathlib import Path


class VectorStore:
	def __init__(self, persist_directory: Path, collection_name: str, embedding_dimension: Optional[int] = None) -> None:
		self.client = Client(Settings(allow_reset=True, is_persistent=True, persist_directory=str(persist_directory)))
		self.collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
		self.embedding_dimension = embedding_dimension

	def add(self, ids: List[str], embeddings: Optional[List[List[float]]] = None, metadatas: Optional[List[Dict[str, Any]]] = None, documents: Optional[List[str]] = None) -> None:
		kwargs: Dict[str, Any] = {"ids": ids}
		if embeddings is not None:
			kwargs["embeddings"] = embeddings
		if metadatas is not None:
			kwargs["metadatas"] = metadatas
		if documents is not None:
			kwargs["documents"] = documents
		self.collection.add(**kwargs)

	def query(self, query_embeddings: List[List[float]], n_results: int = 5, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		kwargs: Dict[str, Any] = {"query_embeddings": query_embeddings, "n_results": n_results}
		if where:
			kwargs["where"] = where
		return self.collection.query(**kwargs)

	def delete(self, ids: List[str]) -> None:
		self.collection.delete(ids=ids)
