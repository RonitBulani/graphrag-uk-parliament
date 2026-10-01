# Filter ParlSpeech V2 down to Brexit-related UK parliamentary speeches (2015-2020)

import pandas as pd
import zipfile
import re
from pathlib import Path


# =============================================================================
# PREPROCESSING HELPERS
# =============================================================================

# strip honorifics so "Mr Smith" and "Smith" map to the same person
_HONORIFIC = re.compile(
    r"^(?:The\s+)?(?:Right\s+Hon(?:ourable)?\.?\s+)?"
    r"(?:Mr\.?|Mrs\.?|Ms\.?|Miss\.?|Dr\.?|Prof\.?|Sir|Dame|Lord|Lady|Baron(?:ess)?|Rev\.?)\.?\s+",
    re.IGNORECASE,
)
_PROCEDURAL_SPEAKERS = {
    "CHAIR", "THE CHAIR", "DEPUTY SPEAKER", "THE DEPUTY SPEAKER",
    "SPEAKER", "THE SPEAKER", "MADAM DEPUTY SPEAKER", "MR DEPUTY SPEAKER",
    "MADAM SPEAKER", "LORD SPEAKER",
}

# remove honorifics from a speaker name
def normalize_speaker(name):
    if not isinstance(name, str):
        return name
    return _HONORIFIC.sub("", name.strip()).strip()


# =============================================================================
# CONFIGURATION
# =============================================================================

_ROOT         = Path(__file__).resolve().parent.parent
INPUT_FILE    = _ROOT / "archive.zip"
OUTPUT_DIR    = _ROOT / "data"
CHUNK_SIZE    = 50_000
DATE_START    = "2015-01-01"
DATE_END      = "2020-12-31"
KEEP_COLS     = ["date", "agenda", "speechnumber", "speaker", "party", "text"]
FILES_TO_READ = ["df_HoC_2000s.csv", "df_HoC_miniDebate.csv"]

# regex patterns to flag Brexit speeches
BREXIT_KEYWORDS = [
    r"\bbrexit\b",
    r"\beu\sreferen",
    r"\bleave\s(?:the\s)?eu\b",
    r"\bremain\s(?:in\s)?(?:the\s)?eu\b",
    r"\beuropean\sunion\b",
    r"\barticle\s50\b",
    r"\bwithdrawal\sagreement\b",
    r"\bcustoms\sunion\b",
    r"\bsingle\smarket\b",
    r"\bbackstop\b",
    r"\bnorthern\sireland\sprotocol\b",
    r"\bno[- ]deal\b",
    r"\btake\sback\scontrol\b",
    r"\bglobal\sbritain\b",
    r"\bfreedom\sof\smovement\b",
    r"\beu\s(?:citizens?|nationals?)\b",
    r"\bchequers\b",
    r"\blancaster\shouse\b",
]
BREXIT_PATTERN = re.compile("|".join(BREXIT_KEYWORDS), re.IGNORECASE)


# =============================================================================
# LOADING DATA
# =============================================================================

print("\nLoading & filtering ParlSpeech")

brexit_frames = []

# read in chunks so the full CSV never loads into memory at once
with zipfile.ZipFile(INPUT_FILE, "r") as z:
    for csv_name in FILES_TO_READ:
        print(f"\n  Processing {csv_name}")
        with z.open(csv_name) as f:
            chunk_num = 0
            for chunk in pd.read_csv(f, low_memory=False, chunksize=CHUNK_SIZE):
                chunk_num += 1
                chunk = chunk[[c for c in KEEP_COLS if c in chunk.columns]]
                # apply keyword + date-range filter
                mask_keywords = chunk["text"].astype(str).str.contains(BREXIT_PATTERN, na=False)
                dates     = pd.to_datetime(chunk["date"], errors="coerce")
                mask_date = (dates >= DATE_START) & (dates <= DATE_END)
                matched   = chunk[mask_keywords & mask_date]
                if len(matched) > 0:
                    brexit_frames.append(matched)
                print(f"    Chunk {chunk_num}: scanned {len(chunk):,} rows, "
                      f"found {len(matched):,} Brexit speeches")

print(f"\n  Combining all matched rows")
df_brexit = pd.concat(brexit_frames, ignore_index=True)
print(f"  Total before cleaning: {len(df_brexit):,} speeches")


# =============================================================================
# CLEANING
# =============================================================================

print("\nCleaning")

# normalise speaker names
df_brexit["speaker"] = df_brexit["speaker"].apply(normalize_speaker)

# drop procedural speakers (chair, speaker, etc.)
before = len(df_brexit)
df_brexit = df_brexit[~df_brexit["speaker"].isin(_PROCEDURAL_SPEAKERS)]
print(f"  Removed {before - len(df_brexit):,} procedural-role speeches")

# drop very short speeches
lengths = df_brexit["text"].astype(str).str.split().str.len()
before  = len(df_brexit)
df_brexit = df_brexit[lengths >= 30]
print(f"  Removed {before - len(df_brexit):,} speeches with < 30 words")

# drop exact duplicates
before    = len(df_brexit)
df_brexit = df_brexit.drop_duplicates(subset=["text"])
print(f"  Removed {before - len(df_brexit):,} duplicate speeches")

# tidy whitespace in text
df_brexit["text"] = (
    df_brexit["text"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
)
df_brexit = df_brexit.reset_index(drop=True)


# =============================================================================
# DATASET SUMMARY
# =============================================================================

print("\nDataset summary")
print(f"  Total speeches:    {len(df_brexit):,}")

word_counts = df_brexit["text"].str.split().str.len()
print(f"  Total words:       {word_counts.sum():,}")
print(f"  Avg words/speech:  {word_counts.mean():.0f}")

print("\n  Party breakdown:")
for line in df_brexit["party"].value_counts().head(10).to_string().split("\n"):
    print(f"    {line}")

print("\n  Year breakdown:")
df_brexit["year"] = pd.to_datetime(df_brexit["date"]).dt.year
for line in df_brexit["year"].value_counts().sort_index().to_string().split("\n"):
    print(f"    {line}")
df_brexit.drop(columns=["year"], inplace=True)


# =============================================================================
# SAVING OUTPUT
# =============================================================================

output_dir  = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "parlspeech_brexit_filtered.csv"
df_brexit.to_csv(output_path, index=False)
print(f"\n  + {output_path}  ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

# save a tiny sample for quick inspection
sample_path = output_dir / "parlspeech_brexit_sample_50.csv"
df_brexit.sample(n=min(50, len(df_brexit)), random_state=42).to_csv(sample_path, index=False)
print(f"  + {sample_path}  (50-row sample)")