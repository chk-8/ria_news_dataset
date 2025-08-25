from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


class GSDMM:
	def __init__(self, num_topics: int = 20, alpha: float = 0.1, beta: float = 0.1, max_iter: int = 50):
		self.K = num_topics
		self.alpha = alpha
		self.beta = beta
		self.max_iter = max_iter
		self.cluster_word_counts: List[Counter] = [Counter() for _ in range(self.K)]
		self.cluster_doc_counts: List[int] = [0 for _ in range(self.K)]
		self.vocab: Dict[str, int] = {}

	def fit(self, docs: List[List[str]]) -> List[int]:
		for doc in docs:
			for w in doc:
				self.vocab.setdefault(w, 0)
				self.vocab[w] += 1
		V = len(self.vocab)
		assignments = [random.randrange(self.K) for _ in docs]
		for d, z in enumerate(assignments):
			self.cluster_doc_counts[z] += 1
			for w in docs[d]:
				self.cluster_word_counts[z][w] += 1
		for _ in range(self.max_iter):
			changed = 0
			for d, doc in enumerate(docs):
				z = assignments[d]
				self.cluster_doc_counts[z] -= 1
				for w in doc:
					self.cluster_word_counts[z][w] -= 1
					if self.cluster_word_counts[z][w] == 0:
						del self.cluster_word_counts[z][w]
				probs = []
				for k in range(self.K):
					term1 = (self.cluster_doc_counts[k] + self.alpha)
					term2 = 1.0
					Nk = sum(self.cluster_word_counts[k].values())
					for w in doc:
						term2 *= (self.cluster_word_counts[k].get(w, 0) + self.beta) / (Nk + V * self.beta)
					probs.append(term1 * term2)
				total = sum(probs)
				if total == 0:
					probs = [1.0 / self.K] * self.K
				else:
					probs = [p / total for p in probs]
				# sample new cluster
				r = random.random()
				cum = 0.0
				new_z = 0
				for k, p in enumerate(probs):
					cum += p
					if r <= cum:
						new_z = k
						break
				assignments[d] = new_z
				self.cluster_doc_counts[new_z] += 1
				for w in doc:
					self.cluster_word_counts[new_z][w] += 1
				if new_z != z:
					changed += 1
			if changed == 0:
				break
		return assignments

	def top_words(self, k: int, n: int = 10) -> List[Tuple[str, int]]:
		return self.cluster_word_counts[k].most_common(n)


def fit_topics(documents: List[str], num_topics: int = 20, max_iter: int = 50) -> Tuple[GSDMM, List[int], List[float]]:
	docs_tokens = [doc.split() for doc in documents]
	model = GSDMM(num_topics=num_topics, max_iter=max_iter)
	assignments = model.fit(docs_tokens)
	# naive "probability": relative mass of assigned cluster
	cluster_sizes = Counter(assignments)
	probs = [cluster_sizes[a] / max(1, len(assignments)) for a in assignments]
	return model, assignments, probs

