import re
from typing import List, Tuple

from pymystem3 import Mystem
from razdel import tokenize


_mystem = None


RUSSIAN_STOPWORDS = set(
	[
		"и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она", "так", "его", "но", "да",
		"ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет",
		"о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него",
		"до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они", "тут",
		"где", "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз",
		"тоже", "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем", "ним",
		"здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "кажется", "сейчас", "были", "куда", "зачем", "всех",
		"никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть", "после", "над", "больше", "тот", "через", "эти",
		"нас", "про", "всего", "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой",
		"перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им", "более", "всегда", "конечно", "всю", "между",
	]
)


def normalize_text(text: str) -> str:
	if text is None:
		return ""
	# unify quotes and dashes
	text = text.replace("\u2014", "-").replace("\u2013", "-").replace("\u00ab", '"').replace("\u00bb", '"')
	text = text.lower()
	# remove control chars
	text = re.sub(r"[\u0000-\u001f]", " ", text)
	# keep letters, digits, spaces, hyphens
	text = re.sub(r"[^\w\s-]", " ", text, flags=re.UNICODE)
	# collapse spaces
	text = re.sub(r"\s+", " ", text).strip()
	return text


def _ensure_models() -> None:
	global _mystem
	if _mystem is None:
		_mystem = Mystem()


def tokenize_words(text: str) -> List[str]:
	return [t.text for t in tokenize(text)]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
	_ensure_models()
	joined = " ".join(tokens)
	mystem_lemmas = _mystem.lemmatize(joined)
	return [l.strip() for l in mystem_lemmas if l.strip()]


def filter_tokens(tokens: List[str]) -> List[str]:
	return [t for t in tokens if t not in RUSSIAN_STOPWORDS and any(ch.isalpha() for ch in t)]


def preprocess_title(text: str) -> Tuple[str, List[str]]:
	"""Return whitespace-joined lemmas and list of lemmas."""
	norm = normalize_text(text)
	toks = tokenize_words(norm)
	lemmas = lemmatize_tokens(toks)
	filtered = filter_tokens(lemmas)
	return " ".join(filtered), filtered