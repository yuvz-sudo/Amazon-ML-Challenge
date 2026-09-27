"""Entry point — generates both output files."""
import argparse
import sys
import os

# Ensure the parent of src/ is on sys.path so `from src.xxx` imports work
# regardless of which directory the user invokes the script from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument("--train-dir", required=True, help="Path to training data directory")
    parser.add_argument("--test-dir", required=True, help="Path to test data directory")
    parser.add_argument("--output-dir", required=True, help="Path to output directory")
    args = parser.parse_args()

    run_pipeline(args.train_dir, args.test_dir, args.output_dir)


if __name__ == "__main__":
    main()
