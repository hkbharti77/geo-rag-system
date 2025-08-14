from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from shapely.geometry import box

from .embedding_manager import EmbeddingManager
from .vector_store import VectorStore


@dataclass
class RetrievalEngine:
	text_store: VectorStore
	image_store: Optional[VectorStore]
	embeddings: EmbeddingManager

	def semantic_search(self, query: str, n_results: int = 5, bbox: Optional[Tuple[float, float, float, float]] = None) -> Dict[str, Any]:
		query_emb = self.embeddings.embed_texts([query])
		where: Optional[Dict[str, Any]] = None
		if bbox is not None:
			minx, miny, maxx, maxy = bbox
			where = {
				"$and": [
					{"minx": {"$lte": maxx}},
					{"maxx": {"$gte": minx}},
					{"miny": {"$lte": maxy}},
					{"maxy": {"$gte": miny}}
				]
			}
		return self.text_store.query(query_embeddings=query_emb, n_results=n_results, where=where)

	def image_search(self, image_embeddings: List[List[float]], n_results: int = 5, bbox: Optional[Tuple[float, float, float, float]] = None) -> Dict[str, Any]:
		if self.image_store is None:
			return {"ids": [], "distances": [], "metadatas": []}
		where: Optional[Dict[str, Any]] = None
		if bbox is not None:
			minx, miny, maxx, maxy = bbox
			where = {
				"$and": [
					{"minx": {"$lte": maxx}},
					{"maxx": {"$gte": minx}},
					{"miny": {"$lte": maxy}},
					{"maxy": {"$gte": miny}}
				]
			}
		return self.image_store.query(query_embeddings=image_embeddings, n_results=n_results, where=where)
