from dataclasses import dataclass
from typing import List
import os

try:
    from sentence_transformers import SentenceTransformer
    from transformers import CLIPProcessor, CLIPModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from PIL import Image
import torch

# Import simple fallback
from .simple_embedding import SimpleEmbeddingManager


@dataclass
class EmbeddingManager:
	text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
	image_model_name: str = "openai/clip-vit-base-patch32"

	def __post_init__(self) -> None:
		# Check if transformers are available
		if not TRANSFORMERS_AVAILABLE:
			# Use simple fallback if transformers are not available
			self.simple_manager = SimpleEmbeddingManager()
			self.text_model = None
			self.clip_model = None
			self.clip_processor = None
			return
		
		# Handle device placement for PyTorch models
		device = "cuda" if torch.cuda.is_available() else "cpu"
		
		# Set environment variables to avoid meta tensor issues and network timeouts
		os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
		os.environ["TOKENIZERS_PARALLELISM"] = "false"
		os.environ["HF_HUB_OFFLINE"] = "0"  # Allow online downloads
		
		# Initialize text model with proper device handling and network error handling
		try:
			self.text_model = SentenceTransformer(self.text_model_name, device=device)
		except Exception as e:
			# Handle network errors and other issues
			if "MaxRetryError" in str(e) or "NameResolutionError" in str(e) or "getaddrinfo failed" in str(e):
				# Network connectivity issue - use simple fallback
				print(f"Warning: Network error - using simple embedding fallback. Error: {str(e)}")
				self.simple_manager = SimpleEmbeddingManager()
				self.text_model = None
				self.clip_model = None
				self.clip_processor = None
				return
			elif "NotImplementedError" in str(e) or "RuntimeError" in str(e):
				# Meta tensor issues - try fallback approaches
				try:
					self.text_model = SentenceTransformer(self.text_model_name)
					if hasattr(self.text_model, 'to_empty'):
						self.text_model.to_empty(device=device)
					else:
						self.text_model.to(device)
				except Exception as e2:
					# Final fallback - use CPU only
					self.text_model = SentenceTransformer(self.text_model_name, device="cpu")
			else:
				# Other errors
				raise e
		
		# Initialize CLIP model with proper device handling
		try:
			self.clip_model = CLIPModel.from_pretrained(self.image_model_name)
			self.clip_model.to(device)
		except Exception as e:
			# Handle network errors and other issues
			if "MaxRetryError" in str(e) or "NameResolutionError" in str(e) or "getaddrinfo failed" in str(e):
				# Network connectivity issue - skip CLIP model for now
				print(f"Warning: Cannot download CLIP model due to network issues. Image features will be disabled. Error: {str(e)}")
				self.clip_model = None
				self.clip_processor = None
			elif "NotImplementedError" in str(e) or "RuntimeError" in str(e):
				# Meta tensor issues - try fallback approaches
				try:
					self.clip_model = CLIPModel.from_pretrained(self.image_model_name)
					if hasattr(self.clip_model, 'to_empty'):
						self.clip_model.to_empty(device=device)
					else:
						self.clip_model.to(device)
				except Exception as e2:
					# Final fallback - use CPU only
					self.clip_model = CLIPModel.from_pretrained(self.image_model_name)
					self.clip_model.to("cpu")
			else:
				# Other errors
				raise e
		
		# Initialize CLIP processor if model was loaded successfully
		if self.clip_model is not None:
			try:
				self.clip_processor = CLIPProcessor.from_pretrained(self.image_model_name)
			except Exception as e:
				print(f"Warning: Cannot download CLIP processor. Image features will be disabled. Error: {str(e)}")
				self.clip_processor = None

	def embed_texts(self, texts: List[str]) -> List[List[float]]:
		if self.text_model is None and hasattr(self, 'simple_manager'):
			# Use simple fallback
			return self.simple_manager.embed_texts(texts)
		elif self.text_model is None:
			raise Exception("No embedding model available")
		else:
			# Use the full model
			embeddings = self.text_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
			return embeddings.tolist()

	def embed_images(self, images: List[Image.Image]) -> List[List[float]]:
		if self.clip_model is None or self.clip_processor is None:
			raise Exception("CLIP model not available. Image embedding is disabled due to network connectivity issues.")
		
		device = "cuda" if torch.cuda.is_available() else "cpu"
		inputs = self.clip_processor(images=images, return_tensors="pt")
		inputs = {k: v.to(device) for k, v in inputs.items()}
		with torch.no_grad():
			image_features = self.clip_model.get_image_features(**inputs)
		image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
		return image_features.cpu().tolist()
