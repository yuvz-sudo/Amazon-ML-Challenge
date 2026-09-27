"""String similarity feature computation for candidate pairs."""
from rapidfuzz import fuzz
from . import config


def compute_pair_features(s1_row, cand_row):
    """Compute similarity features for a single (S1, candidate) pair."""
    name1 = s1_row['name_norm']
    name2 = cand_row['name_norm']
    addr1 = s1_row['addr_norm']
    addr2 = cand_row['addr_norm']

    features = {}

    # Name features
    features['name_token_sort'] = fuzz.token_sort_ratio(name1, name2) / 100.0
    features['name_token_set'] = fuzz.token_set_ratio(name1, name2) / 100.0
    features['name_partial'] = fuzz.partial_ratio(name1, name2) / 100.0
    features['name_ratio'] = fuzz.ratio(name1, name2) / 100.0

    # Address features
    if addr1 and addr2:
        features['addr_token_sort'] = fuzz.token_sort_ratio(addr1, addr2) / 100.0
        features['addr_token_set'] = fuzz.token_set_ratio(addr1, addr2) / 100.0
        features['addr_partial'] = fuzz.partial_ratio(addr1, addr2) / 100.0
    else:
        features['addr_token_sort'] = 0.0
        features['addr_token_set'] = 0.0
        features['addr_partial'] = 0.0

    # Combined weighted score
    features['combined_score'] = (
        config.WEIGHT_NAME_TOKEN_SORT * features['name_token_sort'] +
        config.WEIGHT_NAME_TOKEN_SET * features['name_token_set'] +
        config.WEIGHT_NAME_PARTIAL * features['name_partial'] +
        config.WEIGHT_ADDR_TOKEN_SORT * features['addr_token_sort'] +
        config.WEIGHT_ADDR_TOKEN_SET * features['addr_token_set'] +
        config.WEIGHT_ADDR_PARTIAL * features['addr_partial']
    )

    return features
