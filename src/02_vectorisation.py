# Chunk the filtered speeches and embed them into a ChromaDB vector store (runs once, ~25 min)

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import os
import re
import nltk
from nltk.tokenize import sent_tokenize
from pathlib import Path

try:
    sent_tokenize("test")
except LookupError:
    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)


# =============================================================================
# PREPROCESSING HELPERS
# =============================================================================

# strip honorifics so speaker names match the knowledge graph
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

_ROOT           = Path(__file__).resolve().parent.parent
DATA_FILE       = _ROOT / "data" / "parlspeech_brexit_filtered.csv"
VECTORSTORE_DIR = _ROOT / "data" / "vectorstore"

CHUNK_SIZE      = 200
CHUNK_OVERLAP   = 75
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE      = 256
MIN_CHUNK_WORDS = 30


# =============================================================================
# CHUNKING FUNCTION
# =============================================================================

# split text into overlapping chunks, always breaking on sentence boundaries
def chunk_speech(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, min_words=MIN_CHUNK_WORDS):
    sentences     = sent_tokenize(text)
    sent_words    = [s.split() for s in sentences]
    result_chunks = []
    i = 0

    while i < len(sentences):
        current_sents = []
        current_wc    = 0
        j = i

        while j < len(sentences):
            wc = len(sent_words[j])
            if current_wc + wc > chunk_size and current_sents:
                break
            current_sents.append(sentences[j])
            current_wc += wc
            j += 1

        if current_wc >= min_words:
            result_chunks.append(" ".join(current_sents))

        if j == i:
            # single sentence longer than chunk_size -- emit it and move on
            result_chunks.append(sentences[i])
            i += 1
            continue

        # slide back by overlap so consecutive chunks share context
        overlap_wc    = 0
        overlap_start = j
        for k in range(j - 1, i - 1, -1):
            overlap_wc += len(sent_words[k])
            if overlap_wc >= overlap:
                overlap_start = k
                break
        else:
            overlap_start = i + 1

        i = max(overlap_start, i + 1)

    return result_chunks


# =============================================================================
# LOADING DATA
# =============================================================================

print("\nLoading filtered Brexit data")

df = pd.read_csv(DATA_FILE)
print(f"  {len(df):,} speeches loaded")

# apply the same normalisation as step 1
df["speaker"] = df["speaker"].apply(normalize_speaker)
before = len(df)
df = df[~df["speaker"].isin(_PROCEDURAL_SPEAKERS)]
df = df[df["party"].notna()]
print(f"  Dropped {before - len(df):,} procedural / unknown-party rows")


# =============================================================================
# CHUNKING SPEECHES
# =============================================================================

print("\nChunking speeches")

chunks    = []
metadatas = []

for i, row in df.iterrows():
    text = str(row["text"])
    if len(text.split()) < MIN_CHUNK_WORDS:
        continue

    # attach speaker metadata so retrieval can filter later
    meta = {
        "speaker":      str(row.get("speaker", "")),
        "party":        str(row.get("party", "")),
        "date":         str(row.get("date", "")),
        "agenda":       str(row.get("agenda", "")),
        "speech_index": i,
    }
    for chunk_text in chunk_speech(text):
        chunks.append(chunk_text)
        metadatas.append(meta)

    if (i + 1) % 5000 == 0:
        print(f"  {i + 1:,} speeches -> {len(chunks):,} chunks so far")

print(f"  Done -- {len(chunks):,} chunks from {len(df):,} speeches")
print(f"  Avg chunk length: {sum(len(c.split()) for c in chunks) / len(chunks):.0f} words")


# =============================================================================
# EMBEDDING AND STORING
# =============================================================================

print("\nEmbedding & storing in ChromaDB")

print("  Loading embedding model")
model = SentenceTransformer(EMBEDDING_MODEL)

os.makedirs(VECTORSTORE_DIR, exist_ok=True)
client = chromadb.PersistentClient(path=VECTORSTORE_DIR)

# always rebuild the collection from scratch
existing = [c.name for c in client.list_collections()]
if "brexit_speeches" in existing:
    client.delete_collection("brexit_speeches")
    print("  Dropped existing collection (rebuilding)")

collection = client.create_collection(
    name="brexit_speeches",
    metadata={"description": "Brexit speeches -- bge-small-en-v1.5, chunk_size=200, overlap=75"}
)

# encode and upload in batches to keep memory manageable
total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_num in range(total_batches):
    start = batch_num * BATCH_SIZE
    end   = min(start + BATCH_SIZE, len(chunks))

    embeddings = model.encode(chunks[start:end], show_progress_bar=False).tolist()
    collection.add(
        ids=[f"chunk_{j}" for j in range(start, end)],
        documents=chunks[start:end],
        embeddings=embeddings,
        metadatas=metadatas[start:end],
    )

    if (batch_num + 1) % 20 == 0 or (batch_num + 1) == total_batches:
        print(f"  Batch {batch_num + 1}/{total_batches} -- {end:,}/{len(chunks):,} chunks stored")


print(f"\n  + {VECTORSTORE_DIR}  ({collection.count():,} chunks)")
print(f"  model={EMBEDDING_MODEL}, chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")


# =============================================================================
# QUICK SANITY CHECK
# =============================================================================

print(f"\nQuick test -- searching for 'Article 50 withdrawal'")
test_results = collection.query(
    query_texts=["Article 50 withdrawal from the European Union"],
    n_results=3,
)
for i, doc in enumerate(test_results["documents"][0]):
    speaker = test_results["metadatas"][0][i]["speaker"]
    date    = test_results["metadatas"][0][i]["date"]
    print(f"\n  Result {i+1} ({speaker}, {date}):")
    print(f"    {doc[:150]}...")