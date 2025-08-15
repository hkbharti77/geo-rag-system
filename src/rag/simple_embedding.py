"""
Simple embedding fallback for when Hugging Face models are not available
"""

import numpy as np
import hashlib
from typing import List


class SimpleEmbeddingManager:
    """Simple embedding manager that creates deterministic embeddings without downloading models"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Create simple deterministic embeddings based on text hash"""
        embeddings = []
        for text in texts:
            # Create a deterministic embedding based on text hash
            text_hash = hashlib.md5(text.encode()).hexdigest()
            # Use the hash to seed a random number generator
            np.random.seed(int(text_hash[:8], 16))
            # Generate a random embedding
            embedding = np.random.normal(0, 1, self.embedding_dim)
            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding.tolist())
        return embeddings
    
    def embed_images(self, images) -> List[List[float]]:
        """Placeholder for image embeddings"""
        raise NotImplementedError("Image embeddings not available in simple mode")


def create_simple_embedding_manager():
    """Factory function to create a simple embedding manager"""
    return SimpleEmbeddingManager()
