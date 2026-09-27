"""Configuration: paths, constants, and hyperparameters."""
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "..", "..", "student_resource", "dataset", "train")
TEST_DIR = os.path.join(BASE_DIR, "..", "..", "student_resource", "dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "..", "output")

# --- Reproducibility ---
SEED = 42

# --- TF-IDF Blocking ---
TFIDF_ANALYZER = "char_wb"
TFIDF_NGRAM_RANGE = (3, 5)
TFIDF_MAX_FEATURES = 100_000
TFIDF_SUBLINEAR_TF = True

BLOCKING_TOP_K = 20
BLOCKING_MIN_SIMILARITY = 0.05
BLOCKING_BATCH_SIZE = 10_000

# --- Matching ---
# This is overridden after validation tuning
MATCH_THRESHOLD = 0.50

# --- Feature Weights (combined score) ---
WEIGHT_NAME_TOKEN_SORT = 0.40
WEIGHT_NAME_TOKEN_SET = 0.15
WEIGHT_NAME_PARTIAL = 0.10
WEIGHT_ADDR_TOKEN_SORT = 0.20
WEIGHT_ADDR_TOKEN_SET = 0.10
WEIGHT_ADDR_PARTIAL = 0.05

# --- Validation ---
VAL_FRACTION = 0.05
