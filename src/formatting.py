import json
import random
from typing import List, Dict, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Semantic Brain of an autonomous AI engineer.
Your role is to route user queries to the correct tool or answer directly.

OUTPUT RULES:
1. If the user asks a question you can answer with general knowledge, return status="complete".
2. If the user asks for a specific action (search, file edit, debug), return status="running" and choose the tool.
3. If the request is ambiguous or impossible, return status="running" and use the 'ask_human' tool.
4. Output STRICT JSON only. No markdown, no yapping."""

def validate_entry(entry: Dict) -> bool:
    """
    Validates against the AgentOutput schema.
    """

    if "user_query" not in entry or "output" not in entry:
        logger.warning(f"Missing required fields in entry: {entry.keys()}")
        return False
    
    output = entry["output"]
    if "status" not in output:
        logger.warning(f"Missing 'status' in output: {output}")
        return False
    
    # Validate status values
    if output["status"] not in ["complete", "running"]:
        logger.warning(f"Invalid status: {output['status']}")
        return False
    
    # If running, should have tool_name
    if output["status"] == "running" and "tool_name" not in output:
        logger.warning(f"Status 'running' but no tool_name: {output}")
        return False
    
    return True

def format_llama3(entry: Dict, add_bos: bool = False) -> str:
    """
    Converts a raw entry into Llama-3 Instruct format.
    
    Args:
        entry: Dictionary with 'user_query' and 'output' keys
        add_bos: Whether to add <|begin_of_text|> token (usually handled by tokenizer)
    
    Returns:
        Formatted string ready for training
    """
    user_query = entry["user_query"].strip()
    
    # Serialize output with consistent formatting (no extra spaces)
    target_json = json.dumps(entry["output"], separators=(',', ':'), ensure_ascii=False)
    
    # Build the conversation
    text = ""
    if add_bos:
        text += "<|begin_of_text|>"
    
    text += f"""<|start_header_id|>system<|end_header_id|>

    {SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    {target_json}<|eot_id|>"""
    
    return text

def load_and_validate_data(input_path: str) -> List[Dict]:
    """Load JSONL and validate entries."""
    data = []
    invalid_count = 0
    
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                if validate_entry(entry):
                    data.append(entry)
                else:
                    invalid_count += 1
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error on line {i}: {e}")
                invalid_count += 1
    
    logger.info(f"Loaded {len(data)} valid examples, skipped {invalid_count} invalid entries")
    return data

def stratified_split(data: List[Dict], train_ratio: float = 0.9, 
                     seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Split data with stratification by status to ensure balanced representation.
    """
    random.seed(seed)
    
    # Group by status
    complete_items = [d for d in data if d["output"]["status"] == "complete"]
    running_items = [d for d in data if d["output"]["status"] == "running"]
    
    random.shuffle(complete_items)
    random.shuffle(running_items)
    
    # Split each group
    complete_split = int(len(complete_items) * train_ratio)
    running_split = int(len(running_items) * train_ratio)
    
    train_data = complete_items[:complete_split] + running_items[:running_split]
    test_data = complete_items[complete_split:] + running_items[running_split:]
    
    # Shuffle the combined sets
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    logger.info(f"Train set: {len(train_data)} examples "
                f"({sum(1 for d in train_data if d['output']['status'] == 'complete')} complete, "
                f"{sum(1 for d in train_data if d['output']['status'] == 'running')} running)")
    logger.info(f"Test set: {len(test_data)} examples "
                f"({sum(1 for d in test_data if d['output']['status'] == 'complete')} complete, "
                f"{sum(1 for d in test_data if d['output']['status'] == 'running')} running)")
    
    return train_data, test_data

def save_dataset(data: List[Dict], output_path: str, add_bos: bool = False):
    """Format and save dataset to JSONL."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            formatted = {"text": format_llama3(item, add_bos=add_bos)}
            f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(data)} examples to {output_path}")

def print_sample(data: List[Dict], n: int = 2):
    """Print sample formatted examples for inspection."""
    logger.info(f"\n{'='*80}\nSample Formatted Examples:\n{'='*80}")
    for i, item in enumerate(random.sample(data, min(n, len(data))), 1):
        logger.info(f"\n--- Sample {i} ---")
        logger.info(format_llama3(item))
        logger.info(f"{'='*80}")

def main():
    # Paths
    input_path = "data/raw/router_train_v1.jsonl"
    train_path = "data/processed/train.jsonl"
    test_path = "data/processed/test.jsonl"
    
    # Configuration
    train_ratio = 0.9
    random_seed = 42
    add_bos_token = False  # Usually handled by tokenizer
    
    # Load and validate
    logger.info("Loading and validating data...")
    data = load_and_validate_data(input_path)
    
    if len(data) == 0:
        logger.error("No valid data found. Exiting.")
        return
    
    # Stratified split
    logger.info("Performing stratified split...")
    train_data, test_data = stratified_split(data, train_ratio, random_seed)
    
    # Show samples before saving
    print_sample(train_data)
    
    # Save datasets
    logger.info("Saving formatted datasets...")
    save_dataset(train_data, train_path, add_bos=add_bos_token)
    save_dataset(test_data, test_path, add_bos=add_bos_token)
    
    logger.info("\n✓ Data formatting complete!")
    logger.info(f"  Train: {train_path} ({len(train_data)} examples)")
    logger.info(f"  Test: {test_path} ({len(test_data)} examples)")

if __name__ == "__main__":
    main()