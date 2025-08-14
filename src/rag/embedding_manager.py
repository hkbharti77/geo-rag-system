from dataclasses import dataclass
from typing import List

from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch


@dataclass
class EmbeddingManager:
	text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
	image_model_name: str = "openai/clip-vit-base-patch32"

	def __post_init__(self) -> None:
		self.text_model = SentenceTransformer(self.text_model_name)
		self.clip_model = CLIPModel.from_pretrained(self.image_model_name)
		self.clip_processor = CLIPProcessor.from_pretrained(self.image_model_name)

	def embed_texts(self, texts: List[str]) -> List[List[float]]:
		embeddings = self.text_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
		return embeddings.tolist()

	def embed_images(self, images: List[Image.Image]) -> List[List[float]]:
		inputs = self.clip_processor(images=images, return_tensors="pt")
		with torch.no_grad():
			image_features = self.clip_model.get_image_features(**inputs)
		image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
		return image_features.cpu().tolist()
