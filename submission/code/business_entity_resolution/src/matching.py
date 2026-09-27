"""Threshold-based matcher that scores candidate pairs."""
import pandas as pd
from tqdm.auto import tqdm

from .features import compute_pair_features
from . import config


def score_candidates(s1_df, s23_df, candidates_dict, threshold=None):
    """
    Score candidate pairs and return matches above threshold.

    Args:
        s1_df: Source 1 dataframe (preprocessed)
        s23_df: Combined Source 2+3 dataframe (preprocessed)
        candidates_dict: {s1_id: [candidate_ids]}
        threshold: minimum combined_score to be considered a match

    Returns:
        matches: dict {s1_id: set of matched ids}
    """
    threshold = threshold or config.MATCH_THRESHOLD

    s1_lookup = s1_df.set_index('entity_id')
    s23_lookup = s23_df.set_index('entity_id')

    matches = {}

    for s1_id in tqdm(candidates_dict, desc="Scoring pairs", leave=False):
        cand_ids = candidates_dict[s1_id]
        matched = set()

        if len(cand_ids) == 0:
            matches[s1_id] = matched
            continue

        s1_row = s1_lookup.loc[s1_id]

        for cand_id in cand_ids:
            if cand_id not in s23_lookup.index:
                continue
            cand_row = s23_lookup.loc[cand_id]

            features = compute_pair_features(s1_row, cand_row)

            if features['combined_score'] >= threshold:
                matched.add(cand_id)

        matches[s1_id] = matched

    return matches
