import sys
from pathlib import Path

# --- Set up Path for Imports ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) 

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.infrastructure import logger
from src.data import (
    load_and_validate_data, 
    stratified_split, 
    save_dataset
)

def run_data_formatting():
    logger.info("--- Starting Data Formatting and Split Pipeline ---")

    # 1. FIND ALL FILES
    jsonl_files = list(RAW_DATA_DIR.glob("*.jsonl"))
    if not jsonl_files:
        logger.error(f"No JSONL files found in {RAW_DATA_DIR}. Please run data generation first.")
        return
    
    logger.info(f"Found {len(jsonl_files)} raw files to process.")

    # 2. AGGREGATE DATA
    all_valid_data = []
    seen_queries = set() # For Deduplication
    duplicates_count = 0

    for input_file in jsonl_files:
        logger.info(f"Reading: {input_file.name}")
        try:
            # NOTE: This function handles Pydantic and Semantic checks internally
            data = load_and_validate_data(input_file, use_full_validation=True)            

            # Add to master list with Deduplication
            for item in data:
                if item.user_query not in seen_queries:
                    seen_queries.add(item.user_query)
                    all_valid_data.append(item)
                else:
                    duplicates_count += 1

        except Exception as e:
            logger.warning(f"Failed to process {input_file.name}: {e}. Skipping file.")
            continue
    
    if not all_valid_data:
        logger.error("No valid data found in any files. Exiting.")
        return

    logger.info(f"Aggregation Complete.")
    logger.info(f"Total Unique Items: {len(all_valid_data)}")
    if duplicates_count > 0:
        logger.info(f"Removed {duplicates_count} duplicate queries.")

    # 3. SPLIT (Stratified Split)
    try:
        # Default ratio is 0.9, seed=42 for reproducibility (already set in the function)
        train_data, test_data = stratified_split(all_valid_data) 
        logger.info(f"Split Complete. Train set: {len(train_data)} examples | Test set: {len(test_data)} examples")
    except Exception as e:
        logger.exception(f"FATAL: Failed during stratified splitting.")
        return

    # 4. FORMAT & SAVE (Creates the Llama-3 ChatML string)
    try:
        train_out = PROCESSED_DATA_DIR / "train.jsonl"
        test_out = PROCESSED_DATA_DIR / "test.jsonl"

        save_dataset(train_data, train_out)
        save_dataset(test_data, test_out)
        
        logger.info(f"\nSaved Train set to: {train_out}")
        logger.info(f"Saved Test set to:  {test_out}")
        logger.info("--- Pipeline Complete. Data is ready for fine-tuning. ---")
        
    except Exception as e:
        logger.exception(f"FATAL: Failed during formatting and saving.")

if __name__ == "__main__":
    run_data_formatting()