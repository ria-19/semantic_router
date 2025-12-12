import json
from typing import List, Union

from src.schemas import TrainingExample
from src.validators import DataValidator
from src.infrastructure import logger 


def save_batch_validated(batch_items: List[Union[dict, TrainingExample]], output_file: str) -> int:
    """
    Validates and saves a batch of TrainingExample objects to a JSONL file.
    
    CRITICAL: Uses model_dump_json(exclude_none=True) to remove null fields 
    from the Discriminated Union, ensuring clean training data.
    """
    # 1. Ensure all items are Pydantic objects first
    validated_pydantic_items = []
    validator = DataValidator() 
    
    for item in batch_items:
        if isinstance(item, dict):
            try:
                # Attempt to coerce raw dicts into the target schema
                item = TrainingExample(**item) 
            except Exception as e:
                logger.error(f"Pydantic Coercion Error on raw dict: {e}")
                continue
        validated_pydantic_items.append(item)

    # 2. Apply Custom/Logic Validation (e.g., checking tool arguments)
    final_valid_items = []
    valid_count = 0

    for item in validated_pydantic_items:
        result = validator.validate_full(item) 
    
    if result.is_valid:
        final_valid_items.append(item)
        valid_count += 1
    else:
        logger.warning(f"Validation failed, skipping item: {result.error_message}")
    
    # 3. Serialize and Save
    with open(output_file, "a", encoding="utf-8") as f:
        for item in final_valid_items:
            json_str = item.model_dump_json(exclude_none=True)
            f.write(json_str + "\n")
            

    logger.info(f"Successfully saved {valid_count}/{len(batch_items)} items to {output_file}.")            
    return valid_count

# Alternative: Batch validation version (more efficient for large batches)
def save_batch_optimized(batch_items: List[TrainingExample], output_file: str) -> int:
    """
    Optimized version that validates entire batch at once.
    Better for large batches (10+ items).
    """
    from src.validators import validate_batch
    
    # Validate all items at once
    valid_items, stats = validate_batch(batch_items, strict=True, log_errors=True)
    
    # Save valid items
    with open(output_file, "a", encoding="utf-8") as f:
        for item in valid_items:
            json_str = item.model_dump_json(exclude_none=True)
            f.write(json_str + "\n")
    
    # Log summary
    if stats['warnings'] > 0:
        print(f"Saved {stats['valid']}/{stats['total']} items ({stats['warnings']} warnings)")
    
    return stats['valid']