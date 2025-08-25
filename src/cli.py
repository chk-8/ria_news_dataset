import json
from pathlib import Path
from typing import Dict, List

import click

from .preprocess import preprocess_title
from .sentiment import SentimentLexicons, load_rusentilex, load_nrc_ru, compute_sentiment
from .topics import fit_topics


@click.group()
def cli():
	pass


@cli.command()
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--rusentilex", "rusentilex_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False)
@click.option("--nrc", "nrc_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False)
@click.option("--out", "out_csv", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--limit", type=int, default=0, help="Limit number of rows for quick runs")
def process(input_path: Path, rusentilex_path: Path | None, nrc_path: Path | None, out_csv: Path, limit: int):
	"""Process RIA dataset jsonlines into enriched CSV."""
	records: List[Dict] = []
	with input_path.open("r", encoding="utf-8") as f:
		for i, line in enumerate(f):
			obj = json.loads(line)
			title = obj.get("title", "")
			lemmas_str, lemmas = preprocess_title(title)
			rec = {
				"title": title,
				"lemmas": lemmas_str,
			}
			records.append(rec)
			if limit and i + 1 >= limit:
				break

	# sentiment
	lex = None
	if rusentilex_path or nrc_path:
		ru = load_rusentilex(str(rusentilex_path)) if rusentilex_path else {}
		nrc_pol, nrc_emo = load_nrc_ru(str(nrc_path)) if nrc_path else ({}, {})
		lex = SentimentLexicons(ru_senti_lex=ru, nrc_polarity=nrc_pol, nrc_emotions=nrc_emo)
		for rec in records:
			res = compute_sentiment(rec["lemmas"].split(), lex)
			rec.update({
				"sentiment": res["sentiment"],
				"pos_count": res["pos_count"],
				"neg_count": res["neg_count"],
				"max_emotion": res["max_emotion"],
			})

	# topics
	docs = [r["lemmas"] for r in records]
	model, topics, probs = fit_topics(docs)
	for rec, t, p in zip(records, topics, probs):
		rec["topic_id"] = int(t)
		rec["topic_prob"] = float(p) if p is not None else None

	# save without pandas to avoid 3.13 wheels
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	import csv as _csv
	with out_csv.open("w", encoding="utf-8", newline="") as wf:
		if records:
			fieldnames = list(records[0].keys())
			writer = _csv.DictWriter(wf, fieldnames=fieldnames)
			writer.writeheader()
			for rec in records:
				writer.writerow(rec)
	click.echo(f"Saved {len(records)} rows to {out_csv}")


if __name__ == "__main__":
	cli()

