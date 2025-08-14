from dataclasses import dataclass
from typing import List, Dict, Any

from .retrieval_engine import RetrievalEngine


@dataclass
class GenerationPipeline:
	retrieval: RetrievalEngine

	def assemble_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
		res = self.retrieval.semantic_search(query, n_results=top_k)
		items = []
		for i, meta in enumerate(res.get("metadatas", [[]])[0]):
			items.append({
				"rank": i + 1,
				"metadata": meta,
				"distance": float(res.get("distances", [[None]])[0][i]) if res.get("distances") else None,
				"id": res.get("ids", [[None]])[0][i]
			})
		return {"query": query, "retrieved": items}
