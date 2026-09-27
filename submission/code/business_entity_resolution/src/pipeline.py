"""End-to-end pipeline orchestration."""
import os
import numpy as np
import pandas as pd

from . import config
from .preprocessing import preprocess_df
from .blocking import generate_candidates
from .matching import score_candidates
from .evaluation import compute_f05_score, parse_ground_truth


def load_data(train_dir, test_dir):
    """Load all TSV files."""
    data = {}
    data['train_s1'] = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t")
    data['train_s2'] = pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t")
    data['train_s3'] = pd.read_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t")
    data['train_gt'] = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t")
    data['test_s1'] = pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t")
    data['test_s2'] = pd.read_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t")
    data['test_s3'] = pd.read_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t")
    return data


def tune_threshold(val_pair_scores, val_gt_dict):
    """Find the threshold that maximizes F_0.5 on validation data."""
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    best_f05 = 0
    best_threshold = 0.50

    for thresh in thresholds:
        predictions = {}
        for s1_id in val_pair_scores:
            matched = {cid for cid, score in val_pair_scores[s1_id].items()
                       if score >= thresh}
            predictions[s1_id] = matched

        f05 = compute_f05_score(predictions, val_gt_dict)
        print(f"  threshold={thresh:.2f}  F_0.5={f05:.4f}")

        if f05 > best_f05:
            best_f05 = f05
            best_threshold = thresh

    return best_threshold, best_f05


def write_outputs(test_s1, test_matches, test_candidates, output_dir):
    """Write both output TSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # matching_results.tsv
    rows = []
    for s1_id in test_s1['entity_id']:
        matched = test_matches.get(s1_id, set())
        rows.append({
            'source1_entity_id': s1_id,
            'matched_entity_ids': ','.join(sorted(matched)) if matched else ''
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "matching_results.tsv"), sep="\t", index=False
    )

    # candidate_pairs.tsv
    rows = []
    for s1_id in test_s1['entity_id']:
        cands = test_candidates.get(s1_id, [])
        rows.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': ','.join(sorted(cands)) if cands else ''
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "candidate_pairs.tsv"), sep="\t", index=False
    )


def run_pipeline(train_dir, test_dir, output_dir):
    """Execute the full entity resolution pipeline."""
    np.random.seed(config.SEED)

    # 1. Load
    print("Loading data...")
    data = load_data(train_dir, test_dir)

    # 2. Preprocess
    print("Preprocessing...")
    for key in data:
        if key != 'train_gt':
            data[key] = preprocess_df(data[key])

    # 3. Validation split & threshold tuning
    print("Tuning threshold on validation split...")
    val_mask = np.random.rand(len(data['train_gt'])) < config.VAL_FRACTION
    val_gt = data['train_gt'][val_mask].reset_index(drop=True)
    val_gt_dict = parse_ground_truth(val_gt)
    val_s1_ids = set(val_gt['source1_entity_id'])

    val_s1_df = data['train_s1'][data['train_s1']['entity_id'].isin(val_s1_ids)].reset_index(drop=True)
    val_candidates = generate_candidates(val_s1_df, data['train_s2'], data['train_s3'])

    # Score validation pairs
    from .features import compute_pair_features
    train_s23 = pd.concat([data['train_s2'], data['train_s3']], ignore_index=True)
    s1_lookup = val_s1_df.set_index('entity_id')
    s23_lookup = train_s23.set_index('entity_id')

    val_pair_scores = {}
    for s1_id in val_candidates:
        scores = {}
        if s1_id not in s1_lookup.index:
            val_pair_scores[s1_id] = scores
            continue
        s1_row = s1_lookup.loc[s1_id]
        for cand_id in val_candidates[s1_id]:
            if cand_id not in s23_lookup.index:
                continue
            features = compute_pair_features(s1_row, s23_lookup.loc[cand_id])
            scores[cand_id] = features['combined_score']
        val_pair_scores[s1_id] = scores

    best_threshold, best_f05 = tune_threshold(val_pair_scores, val_gt_dict)
    print(f"Best threshold: {best_threshold} (F_0.5={best_f05:.4f})")

    # 4. Test blocking
    print("\nGenerating test candidates...")
    test_candidates = generate_candidates(data['test_s1'], data['test_s2'], data['test_s3'])

    # 5. Test matching
    print("Matching test candidates...")
    test_s23 = pd.concat([data['test_s2'], data['test_s3']], ignore_index=True)
    test_matches = score_candidates(data['test_s1'], test_s23, test_candidates,
                                    threshold=best_threshold)

    # 6. Write outputs
    print("Writing outputs...")
    write_outputs(data['test_s1'], test_matches, test_candidates, output_dir)
    print(f"Done! Files written to {output_dir}/")
