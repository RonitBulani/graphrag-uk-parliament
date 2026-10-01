# Data

The full experiment uses UK parliamentary speeches from the ParlSpeech V2 source dataset, filtered to Brexit-related House of Commons material between 2015 and 2020.

The full source archive and full derived corpus are **not committed to this repository**. This avoids redistributing a large third-party dataset and keeps the portfolio repository manageable.

## Included files

- `questions.json` — 50 evaluation questions labelled 1-hop, 2-hop or 3-hop
- `parlspeech_brexit_sample_50.csv` — a small sample of the filtered corpus for understanding the schema

## Rebuilding the dataset

After obtaining the required source dataset, place the archive in the repository root as:

```text
archive.zip
```

Then run:

```bash
python src/01_data_loading.py
```

The script reads the configured House of Commons CSV files from the archive, filters the date range and Brexit keywords, performs cleaning, and writes the derived filtered corpus into this directory.

Please follow the original dataset provider's licence and attribution requirements when obtaining or redistributing the source data.
