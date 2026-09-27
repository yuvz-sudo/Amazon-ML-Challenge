# Business Entity Resolution — Reproduction Guide

## Overview

This pipeline resolves business entities across 3 independent data sources using TF-IDF blocking + multi-feature string similarity matching, optimized for F₀.₅.

## Prerequisites

- Python 3.10+
- ~32 GB RAM (recommended for full dataset)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Directory Structure

```
src/
├── config.py           # Paths, constants, hyperparameters
├── preprocessing.py    # Text normalization (names & addresses)
├── blocking.py         # TF-IDF candidate generation (country-partitioned)
├── features.py         # String similarity feature computation
├── matching.py         # Threshold-based matcher
├── evaluation.py       # F₀.₅ scoring utilities
├── pipeline.py         # End-to-end orchestration
└── run.py              # Entry point — generates both output files
```

## Reproduce End-to-End

Place the dataset under `student_resource/dataset/` relative to the project root, then:

```bash
python3 src/run.py \
    --train-dir ../../student_resource/dataset/train \
    --test-dir  ../../student_resource/dataset/test \
    --output-dir ../../output
```

This will:
1. Load and preprocess all sources
2. Run TF-IDF blocking per country → `candidate_pairs.tsv`
3. Score candidate pairs and apply threshold → `matching_results.tsv`

Both files are written to the `--output-dir`.

## Validate Output

```bash
cd ../../student_resource
python3 utils/validate_submission.py \
    --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Expected: `PASS` with exit code 0.

## Key Hyperparameters

| Parameter | Value | Location |
|-----------|-------|----------|
| TF-IDF n-gram range | (3, 5) char_wb | `config.py` |
| TF-IDF max features | 100,000 | `config.py` |
| Blocking top-K | 20 | `config.py` |
| Blocking min similarity | 0.05 | `config.py` |
| Match threshold | Tuned on val set | `config.py` |
| Batch size | 10,000 | `config.py` |
