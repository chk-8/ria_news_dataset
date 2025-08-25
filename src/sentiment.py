from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


NEGATIONS = {"не", "ни", "без", "нет", "неа", "никак"}
INTENSIFIERS = {"очень", "крайне", "супер", "ультра", "чрезвычайно", "ужасно", "дико", "абсолютно"}
DIMINISHERS = {"слегка", "едва", "немного", "чуть", "чуть-чуть"}


@dataclass
class SentimentLexicons:
	ru_senti_lex: Dict[str, float]
	nrc_polarity: Dict[str, float]
	nrc_emotions: Dict[str, Dict[str, int]]


def load_rusentilex(path: str) -> Dict[str, float]:
	lex: Dict[str, float] = {}
	if path.endswith(".json"):
		with open(path, "r", encoding="utf-8") as f:
			obj = json.load(f)
			for k, v in obj.items():
				try:
					lex[k] = float(v)
				except Exception:
					continue
		return lex
	# Fallback: CSV/TSV with word,polarity
	with open(path, "r", encoding="utf-8") as f:
		reader = csv.reader(f, delimiter="," if "," in f.readline() else "\t")
		f.seek(0)
		for row in reader:
			if not row:
				continue
			w = row[0].strip()
			try:
				s = float(row[1])
			except Exception:
				continue
			lex[w] = s
	return lex


def load_nrc_ru(path: str) -> Tuple[Dict[str, float], Dict[str, Dict[str, int]]]:
	polarity: Dict[str, float] = {}
	emotions: Dict[str, Dict[str, int]] = {}
	if path.endswith(".json"):
		with open(path, "r", encoding="utf-8") as f:
			obj = json.load(f)
			for word, feats in obj.items():
				if isinstance(feats, dict):
					# Expect keys like positive, negative, anger, joy, etc.
					pos = float(feats.get("positive", 0))
					neg = float(feats.get("negative", 0))
					polarity[word] = pos - neg
					emo = {k: int(v) for k, v in feats.items() if k not in {"positive", "negative"}}
					emotions[word] = emo
			else:
				pass
		return polarity, emotions
	# Fallback: NRC format word\temotion\t0/1 per line
	with open(path, "r", encoding="utf-8") as f:
		reader = csv.reader(f, delimiter="\t")
		for row in reader:
			if len(row) < 3:
				continue
			w, label, val = row[0].strip(), row[1].strip(), row[2].strip()
			if w not in emotions:
				emotions[w] = {}
			try:
				iv = int(val)
			except Exception:
				iv = 0
			emotions[w][label] = iv
			if label in {"positive", "negative"} and iv:
				polarity[w] = polarity.get(w, 0.0) + (1.0 if label == "positive" else -1.0)
	return polarity, emotions


def compute_sentiment(tokens: List[str], lex: SentimentLexicons) -> Dict[str, object]:
	pos_score = 0.0
	neg_score = 0.0
	emo_counts: Dict[str, int] = {}
	negation_active = False
	intensity_multiplier = 1.0

	for i, tok in enumerate(tokens):
		w = tok.lower()
		if w in NEGATIONS:
			negation_active = True
			continue
		if w in INTENSIFIERS:
			intensity_multiplier *= 1.5
			continue
		if w in DIMINISHERS:
			intensity_multiplier *= 0.7
			continue

		val = 0.0
		if w in lex.ru_senti_lex:
			val += lex.ru_senti_lex[w]
		if w in lex.nrc_polarity:
			val += lex.nrc_polarity[w]

		if val != 0.0:
			if negation_active:
				val = -val
			val *= intensity_multiplier
			if val > 0:
				pos_score += val
			else:
				neg_score += -val

		# emotions
		if w in lex.nrc_emotions:
			for emo, flag in lex.nrc_emotions[w].items():
				if flag:
					emo_counts[emo] = emo_counts.get(emo, 0) + 1

		# reset scope after a content word
		negation_active = False
		intensity_multiplier = 1.0

	polarity = pos_score - neg_score
	max_emo = max(emo_counts, key=emo_counts.get) if emo_counts else None
	return {
		"pos_count": pos_score,
		"neg_count": neg_score,
		"sentiment": polarity,
		"max_emotion": max_emo,
		"emotions": emo_counts,
	}

