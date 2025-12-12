# scripts/audit_data.py

import sys
from pathlib import Path
import os

# --- Set up Path for Imports ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) 

from src.validators import validate_jsonl_file
from src.infrastructure import logger
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def audit_existing_dataset(filepath: Path):
    """
    Run validation on an existing dataset without modifying it.
    This provides a diagnostic report on data quality issues.
    """    
    logger.info(f"\n--- Starting Data Audit on: {filepath.name} ---")
    
    # Check if file exists before proceeding
    if not filepath.exists():
        logger.error(f"File not found at {filepath}. Cannot audit.")
        return

    # Call the core validation logic (which lives in src/validators.py)
    # The function is assumed to return a dictionary of statistics
    stats = validate_jsonl_file(filepath)
    
    logger.info(f"\n--- Audit Results for {filepath.name} ---")
    logger.info(f"  Total Examples Audited: {stats['total']}")
    logger.info(f"  -------------------------------------")
    logger.info(f"  Valid Examples:       {stats['valid']}")
    logger.info(f"  Invalid Examples:     {stats['total'] - stats['valid']}")
    logger.info(f"  -------------------------------------")
    logger.info(f"  % Valid:                {stats['valid']/stats['total']*100:.2f}%")
    logger.info(f"  Parse Errors (JSON/Format): {stats.get('parse_errors', 'N/A')}")
    logger.info(f"  Quality Errors (Thought/Parroting): {stats.get('invalid_quality', 'N/A')}")
    logger.info(f"  Domain Errors (Code/Security): {stats.get('invalid_domain', 'N/A')}")
    logger.info("-------------------------------------\n")


def main():
    """Main entry point to audit the most recent raw and processed files."""
    
    # 1. Audit the most recent RAW file
    raw_files = list(RAW_DATA_DIR.glob("*.jsonl"))
    if raw_files:
        # Sort by creation time to get the newest one
        newest_raw = max(raw_files, key=os.path.getctime)
        audit_existing_dataset(newest_raw)
    else:
        logger.warning(f"No raw data files found in {RAW_DATA_DIR}.")

    # 2. Audit the PROCESSED train file (if it exists)
    train_file = PROCESSED_DATA_DIR / "train.jsonl"
    if train_file.exists():
        audit_existing_dataset(train_file)
    else:
        logger.warning(f"Processed train.jsonl not found in {PROCESSED_DATA_DIR}.")


if __name__ == "__main__":
    main()