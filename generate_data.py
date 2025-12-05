import os
import random
import time
from tqdm import tqdm
from dotenv import load_dotenv

from src import generate_batch, save_batch, logger, log_section, log_metrics
from src.config import INTENT_DISTRIBUTION, DOMAINS, PERSONAS, QUERY_STYLES

def main():
    load_dotenv()

    OUTPUT_FILE = "data/raw/router_train_001.jsonl"
    TARGET_TOTAL = 500 
    BATCH_SIZE = 3

    print("Starting Synthetic Data Generation Pipeline")
    print(f"Target: {TARGET_TOTAL} examples")
    print(f"Output: {OUTPUT_FILE}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    total_loops = (TARGET_TOTAL // BATCH_SIZE) + 1
    saved_total = 0
    
    # Progress Bar
    pbar = tqdm(total=TARGET_TOTAL, desc="   Generators Running", unit="ex", colour="green")

    try:
        start_time = time.time()
        for loop_idx in range(total_loops):
            if saved_total >= TARGET_TOTAL:
                break

            # 1. Weighted Random Selection of Intent
            intent_config = random.choices(
                INTENT_DISTRIBUTION, 
                weights=[x['weight'] for x in INTENT_DISTRIBUTION],
                k=1
            )[0]
            
            # 2. Randomize context
            domain = random.choice(DOMAINS)
            persona = random.choice(PERSONAS)
            
            # 3. Generate
            batch_items = generate_batch(intent_config, domain, persona, BATCH_SIZE)
            
            # 4. Save with validation
            if batch_items:
                valid_count = save_batch(batch_items, OUTPUT_FILE)
                saved_total += valid_count
                pbar.update(valid_count)
                    
                # Log progress every 10 batches
                if (loop_idx + 1) % 10 == 0:
                    print(f"\nProgress: {saved_total}/{TARGET_TOTAL} ({saved_total/TARGET_TOTAL*100:.1f}%)")
            
                # 5. Respect API rate limits
                time.sleep(4)
    except KeyboardInterrupt:
        logger.warning("\nUser interrupted generation process.")
    finally:
        pbar.close()

    # 3. Metrics Section
    duration = time.time() - start_time
    log_section(logger, "GENERATION COMPLETE")
    
    log_metrics(logger, {
        "Total Examples": saved_total,
        "Target": TARGET_TOTAL,
        "Duration": f"{duration:.2f}s",
        "Output File": OUTPUT_FILE,
        "Avg Speed": f"{(saved_total/duration):.2f} ex/sec"
    })

if __name__ == "__main__":
    main()