
from collections import defaultdict


def build_year_date_index(collection, batch_size=5000):
    """
    Build a read-only mapping of years to dates
    already present in ChromaDB.

    Does not modify the database.
    """
    dates_by_year = defaultdict(set)

    for offset in range(0, collection.count(), batch_size):
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"],
        )

        for metadata in batch["metadatas"]:
            date = str(metadata.get("date", ""))

            if (
                len(date) >= 10
                and date[4] == "-"
                and date[7] == "-"
                and date[:4].isdigit()
            ):
                dates_by_year[date[:4]].add(date)

    return {
        year: sorted(dates)
        for year, dates in dates_by_year.items()
    }


def retrieve_year_candidates(
    engine,
    question,
    year,
    dates_by_year,
    n_results=25,
):
    """
    Retrieve and rerank passages belonging to a
    specific year using the original embedding
    model and cross-encoder.
    """
    year = str(year)
    available_dates = dates_by_year.get(year, [])

    if not available_dates:
        return []

    question_embedding = engine.model.encode(
        [question]
    ).tolist()

    results = engine.collection.query(
        query_embeddings=question_embedding,
        n_results=n_results,
        where={
            "date": {
                "$in": available_dates
            }
        },
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        return []

    pairs = [
        [question, document]
        for document in documents
    ]

    scores = engine.reranker.predict(pairs)

    candidates = []

    for document, metadata, distance, score in zip(
        documents,
        metadatas,
        distances,
        scores,
    ):
        # Defensive verification of the actual date.
        if not metadata["date"].startswith(year + "-"):
            continue

        candidates.append({
            "text": document,
            "speaker": metadata["speaker"],
            "party": metadata["party"],
            "date": metadata["date"],
            "agenda": metadata.get("agenda", ""),
            "distance": float(distance),
            "source": f"year_filtered_{year}",
            "rerank_score": float(score),
        })

    return sorted(
        candidates,
        key=lambda c: c["rerank_score"],
        reverse=True,
    )

def select_temporal_evidence(
    original_chunks,
    candidates_by_year,
    requested_years,
    max_chunks=5,
    min_score=-1.5,
):
    """
    Combine original GraphRAG evidence with year-filtered
    candidates while retaining temporal coverage.

    The score threshold is an initial heuristic,
    not a calibrated probability of relevance.
    """

    years = list(dict.fromkeys(
        str(year) for year in requested_years
    ))

    if len(years) > max_chunks:
        raise ValueError(
            "This selection version requires at least "
            "one available slot per requested year."
        )

    # Combine the original and year-filtered candidates.
    pool = list(original_chunks)

    for year in years:
        pool.extend(candidates_by_year.get(year, []))

    # Deduplicate passages, retaining the higher score.
    unique = {}

    for chunk in pool:
        text = chunk.get("text", "").strip()

        if not text:
            continue

        key = (
            chunk.get("speaker"),
            chunk.get("date"),
            text[:160],
        )

        if (
            key not in unique
            or chunk.get("rerank_score", float("-inf"))
            > unique[key].get("rerank_score", float("-inf"))
        ):
            unique[key] = chunk

    # Retain candidates above the chosen score floor.
    eligible = sorted(
        [
            chunk for chunk in unique.values()
            if chunk.get("rerank_score", float("-inf"))
            >= min_score
        ],
        key=lambda c: c["rerank_score"],
        reverse=True,
    )

    selected = []
    selected_ids = set()
    used_speakers = set()

    def passage_id(chunk):
        return (
            chunk.get("speaker"),
            chunk.get("date"),
            chunk["text"].strip()[:160],
        )

    def add_chunk(chunk):
        selected.append(chunk)
        selected_ids.add(passage_id(chunk))
        used_speakers.add(chunk["speaker"])

    # First select the strongest eligible evidence
    # for each requested year.
    for year in years:
        options = [
            chunk for chunk in eligible
            if chunk["date"].startswith(year + "-")
            and passage_id(chunk) not in selected_ids
        ]

        if not options:
            continue

        # Prefer a different speaker when possible.
        choice = next(
            (
                chunk for chunk in options
                if chunk["speaker"] not in used_speakers
            ),
            options[0],
        )

        add_chunk(choice)

    # Fill remaining slots with high-ranking passages
    # from speakers not already represented.
    for chunk in eligible:
        if len(selected) >= max_chunks:
            break

        if (
            passage_id(chunk) not in selected_ids
            and chunk["speaker"] not in used_speakers
        ):
            add_chunk(chunk)

    # If spaces remain, allow additional passages
    # from speakers already represented.
    for chunk in eligible:
        if len(selected) >= max_chunks:
            break

        if passage_id(chunk) not in selected_ids:
            add_chunk(chunk)

    return sorted(
        selected,
        key=lambda c: c["rerank_score"],
        reverse=True,
    )
