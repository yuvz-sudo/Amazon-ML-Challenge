"""F_0.5 scoring and ground truth utilities."""
import numpy as np
import pandas as pd


def compute_f05_score(predictions, ground_truth):
    """
    Compute macro-averaged F_0.5 score.

    predictions: dict {s1_id: set of matched ids}
    ground_truth: dict {s1_id: set of matched ids}
    """
    scores = []
    for s1_id in ground_truth:
        true = ground_truth[s1_id]
        pred = predictions.get(s1_id, set())

        if len(true) == 0 and len(pred) == 0:
            scores.append(1.0)
        elif len(true) == 0 and len(pred) > 0:
            scores.append(0.0)
        elif len(pred) == 0 and len(true) > 0:
            scores.append(0.0)
        else:
            tp = len(pred & true)
            precision = tp / len(pred) if len(pred) > 0 else 0
            recall = tp / len(true) if len(true) > 0 else 0
            if precision + recall == 0:
                scores.append(0.0)
            else:
                f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
                scores.append(f05)

    return np.mean(scores)


def parse_ground_truth(gt_df):
    """Convert ground truth df to dict {s1_id: set of matched ids}."""
    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        if pd.isna(row['matched_entity_ids']):
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(row['matched_entity_ids'].split(','))
    return gt_dict
