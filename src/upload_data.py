import os
from datasets import load_dataset
from huggingface_hub import HfApi, upload_file, create_repo  
from src.logger import logger

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------

USERNAME = "tai-tai-sama"

PROCESSED_REPO = f"{USERNAME}/semantic-router-dataset"
RAW_REPO       = f"{USERNAME}/semantic-router-raw"

PROCESSED_FILES = {
    "train": "data/processed/train.jsonl",
    "test":  "data/processed/test.jsonl"
}

RAW_FILES = [
    "data/raw/router_train_001.jsonl",
    "data/raw/router_train_002.jsonl"
]

# ----------------------------------------------------
# MAIN SCRIPT
# ----------------------------------------------------

def push_processed_dataset():
    logger.info("Loading processed dataset...")

    dataset = load_dataset("json", data_files=PROCESSED_FILES)
    api = HfApi()

    # Create private repo for processed dataset
    try:
        create_repo(PROCESSED_REPO, repo_type="dataset", private=True)
        logger.info(f"Created repo: {PROCESSED_REPO}")
    except Exception:
        logger.info(f"Repo {PROCESSED_REPO} already exists.")

    logger.info(f"Pushing processed dataset to {PROCESSED_REPO}...")
    dataset.push_to_hub(PROCESSED_REPO, private=True)

    logger.info(f"Processed dataset uploaded!")


def push_raw_files():
    logger.info("Uploading RAW JSONL files...")

    # Create dedicated private repo for raw data
    try:
        create_repo(RAW_REPO, repo_type="dataset", private=True)
        logger.info(f"Created repo: {RAW_REPO}")
    except Exception:
        logger.info(f"Repo {RAW_REPO} already exists.")

    logger.info(f"Pushing raw files to {RAW_REPO}...")

    for raw_path in RAW_FILES:
        upload_file(
            path_or_fileobj=raw_path,
            path_in_repo=os.path.basename(raw_path),
            repo_id=RAW_REPO,
            repo_type="dataset"
        )
        logger.info(f"Uploaded {raw_path}")

    logger.info("Raw files uploaded!")


if __name__ == "__main__":
    # push_processed_dataset()
    push_raw_files()
    logger.info("\nAll uploads complete!\n")
    logger.info(f"Load processed dataset with:\n  load_dataset('{PROCESSED_REPO}')")
    logger.info(f"Raw files stored privately in:\n  https://huggingface.co/datasets/{RAW_REPO}")