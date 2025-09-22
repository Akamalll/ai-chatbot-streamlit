import os
from typing import List

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_domain_corpus(domain: str) -> List[str]:
	filename = {
		"edukasi": "edukasi.txt",
		"gizi": "gizi.txt",
		"travel": "travel.txt",
		"produktivitas": "produktivitas.txt",
	}.get(domain.lower(), "edukasi.txt")
	path = os.path.join(DATA_DIR, filename)
	if not os.path.exists(path):
		return []
	with open(path, "r", encoding="utf-8") as f:
		text = f.read()
	# naive split per baris/kalimat singkat
	chunks = [c.strip() for c in text.split("\n") if c.strip()]
	return chunks


class KnowledgeBase:
	def __init__(self, domain: str):
		self.domain = domain
		self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
		self.corpus = _load_domain_corpus(domain)
		self.index, self.vectors = self._build_index(self.corpus)

	def _build_index(self, corpus: List[str]):
		if not corpus:
			return None, None
		embeddings = self.model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
		dim = embeddings.shape[1]
		index = faiss.IndexFlatIP(dim)
		# normalize for cosine similarity using inner product
		embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
		index.add(embeddings_norm.astype(np.float32))
		return index, embeddings_norm

	def search(self, query: str, k: int = 3) -> List[str]:
		if self.index is None or self.vectors is None or not self.corpus:
			return []
		q = self.model.encode([query], convert_to_numpy=True)
		q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
		scores, idxs = self.index.search(q.astype(np.float32), min(k, len(self.corpus)))
		results: List[str] = []
		for i in idxs[0]:
			if 0 <= i < len(self.corpus):
				results.append(self.corpus[i])
		return results



