# "Rossiya Segodnya" news dataset

This repository contains a news dataset presented in the paper:

Daniil Gavrilov, Pavel Kalaidin, and Valentin Malykh. Self-Attentive Model for Headline Generation. 41st European Conference on Information Retrieval, 2019. __[arXiv:1901.07786 [cs.CL]](https://arxiv.org/abs/1901.07786)__

To download the dataset please use a direct [link](https://github.com/RossiyaSegodnya/ria_news_dataset/raw/master/ria.json.gz) or clone the repository using `git lfs`.

## Description

Full dataset contains _1003869_ Russian language news documents from _January, 2010_ to _December, 2014_.

* [`ria_20.json`](./ria_20.json) contains the first 20 news documents from the dataset.

* [`ria_1k.json`](./ria_1k.json) contains the first 1000 news documents from the dataset.

* [`ria.json.gz`](./ria.json.gz) is full GZip'ed dataset.

Dataset format: each row contains a JSON document that consists of two fields: `text` is a document body, while `title` is a news headline.

## License
This data is lisensed by Rossiya Segodnya news agency ([ria.ru](http://ria.ru)) under CC-BY-ND-NC license. The license text could be accessed [here](./LICENSE). The Russian version of the same license could be accessed [here](./LICENSE.ru).

## Misc
If you're using the data in a research please consider citing the mentioned paper:

    @inproceedings{gavrilov2018self,
    	title={Self-Attentive Model for Headline Generation},
    	author={Gavrilov, Daniil and  Kalaidin, Pavel and  Malykh, Valentin},
    	booktitle={Proceedings of the 41st European Conference on Information Retrieval},
    	year={2019}
    }

## Pipeline CLI

Install dependencies (Python 3.10+ recommended):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run processing on a sample subset and save CSV:

```bash
python -m src.cli process --input ria_20.json --out outputs/ria20_enriched.csv --limit 200
```

Provide lexicons for sentiment (RuSentiLex and NRC-RU formats below):

```bash
python -m src.cli process \
  --input ria_1k.json \
  --rusentilex lexicons/rusentilex.json \
  --nrc lexicons/nrc_ru.json \
  --out outputs/ria1k_enriched.csv
```

Lexicon formats:

- `rusentilex.json`: `{ "слово": polarity_float, ... }`
- `nrc_ru.json`: `{ "слово": { "positive": 1|0, "negative": 1|0, "anger": 1|0, ... }, ... }`

Topic modeling uses a lightweight GSDMM implementation suitable for short titles (no GPU required).
