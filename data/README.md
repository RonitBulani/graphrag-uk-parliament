# Data provenance and prepared input

The research dataset is based on **ParlSpeech V2** (Christian Rauh and Jan Schwalbach, 2020), specifically the UK House of Commons corpus.

**Official source:** [ParlSpeech V2, Harvard Dataverse (DOI:10.7910/DVN/L4OAKN)](https://doi.org/10.7910/DVN/L4OAKN). Consult the dataset landing page for the source citation, release files, and applicable terms. The official UK data are distributed as `Corp_HouseOfCommons_V2.rds`; they are **not** distributed as the two CSV files expected by this project's preprocessing script.

## Included public files

- `questions.json` — 50 benchmark questions labelled 1-hop, 2-hop or 3-hop
- `parlspeech_brexit_sample_50.csv` — 50-row sample showing the project's processed-record schema

The full original corpus, full Brexit-filtered dataset, graph and ChromaDB index are not committed to GitHub.

## Reproducing the prepared-data pipeline

`src/01_data_loading.py` reads a **separately prepared** ZIP archive named `archive.zip` from the repository root. It expects the following exact entries inside that ZIP:

```text
df_HoC_2000s.csv
df_HoC_miniDebate.csv
```

The expected input columns are `date`, `agenda`, `speechnumber`, `speaker`, `party`, and `text`. The script filters records whose dates fall between **2015-01-01 and 2020-12-31** and match its Brexit-related keywords, then normalises speakers, removes short/procedural records and exact text duplicates.

**Important limitation:** the date range above is a configured filter window; it does not mean ParlSpeech V2 contains speeches from 2020. The published UK ParlSpeech V2 corpus extends to 2019. The application also limits its supported question period to 2015–2019.

The code in this repository **does not include an RDS-to-CSV conversion or the preparation of these specially named CSV files**. Consequently, downloading the official `.rds` file and renaming it `archive.zip` will not work. A reproducible conversion and schema mapping remain a future improvement.

Once you have verified the provenance and contents of the prepared ZIP archive, run:

```powershell
python src/01_data_loading.py
```

It writes `data/parlspeech_brexit_filtered.csv` and a deterministic 50-row sample. See [Reproducibility](../docs/REPRODUCIBILITY.md) for later indexing, graph-building and evaluation stages. Consult the official dataset provider's citation and licence terms before reuse or redistribution.
