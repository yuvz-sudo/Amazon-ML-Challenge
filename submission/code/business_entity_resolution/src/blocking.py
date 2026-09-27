"""TF-IDF based candidate generation with country partitioning."""
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config


def build_candidates_for_country(s1_df, s2_df, s3_df, country,
                                  top_k=None, batch_size=None):
    """
    Build candidate pairs for a single country using TF-IDF + cosine similarity.

    Returns: dict {s1_id: list of candidate ids from S2/S3}
    """
    top_k = top_k or config.BLOCKING_TOP_K
    batch_size = batch_size or config.BLOCKING_BATCH_SIZE

    # Filter by country
    s1_c = s1_df[s1_df['country'] == country].reset_index(drop=True)
    s2_c = s2_df[s2_df['country'] == country].reset_index(drop=True)
    s3_c = s3_df[s3_df['country'] == country].reset_index(drop=True)

    if len(s1_c) == 0:
        return {}

    # Combine S2 + S3 as the candidate pool
    candidates = pd.concat([s2_c, s3_c], ignore_index=True)

    if len(candidates) == 0:
        return {eid: [] for eid in s1_c['entity_id']}

    print(f"  Country '{country}': S1={len(s1_c):,}, Candidates={len(candidates):,}")

    # Fit TF-IDF on combined corpus
    all_texts = pd.concat([s1_c['combined_text'], candidates['combined_text']],
                          ignore_index=True)

    tfidf = TfidfVectorizer(
        analyzer=config.TFIDF_ANALYZER,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        max_features=config.TFIDF_MAX_FEATURES,
        sublinear_tf=config.TFIDF_SUBLINEAR_TF,
        dtype=np.float32
    )
    tfidf.fit(all_texts)

    # Transform candidates once
    cand_vectors = tfidf.transform(candidates['combined_text'])
    cand_ids = candidates['entity_id'].values

    # Process S1 in batches
    result = {}
    n_batches = (len(s1_c) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc=f"  Blocking ({country})", leave=False):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(s1_c))
        batch_s1 = s1_c.iloc[start:end]

        s1_vectors = tfidf.transform(batch_s1['combined_text'])

        # Compute cosine similarity
        sim_matrix = cosine_similarity(s1_vectors, cand_vectors)

        # Get top-K candidates per S1 entity
        for i in range(len(batch_s1)):
            s1_id = batch_s1.iloc[i]['entity_id']
            sims = sim_matrix[i]

            if len(sims) <= top_k:
                top_indices = np.argsort(sims)[::-1]
            else:
                top_indices = np.argpartition(sims, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

            threshold = config.BLOCKING_MIN_SIMILARITY
            top_cands = [(cand_ids[idx], sims[idx])
                         for idx in top_indices if sims[idx] > threshold]
            result[s1_id] = [cid for cid, _ in top_cands]

    del cand_vectors, tfidf
    return result


def generate_candidates(s1_df, s2_df, s3_df, top_k=None, batch_size=None):
    """Run blocking across all countries."""
    all_countries = sorted(
        set(s1_df['country'].unique())
        | set(s2_df['country'].unique())
        | set(s3_df['country'].unique())
    )
    print(f"Countries: {all_countries}")

    candidates = {}
    for country in all_countries:
        country_cands = build_candidates_for_country(
            s1_df, s2_df, s3_df, country, top_k=top_k, batch_size=batch_size
        )
        candidates.update(country_cands)

    # Ensure every S1 entity has an entry
    for eid in s1_df['entity_id']:
        if eid not in candidates:
            candidates[eid] = []

    return candidates
