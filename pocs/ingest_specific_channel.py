"""
Script to ingest specific Slack channel into DB.
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load env
load_dotenv(project_root / ".env")

from src.sources.slack import SlackSource, SlackConfig, SlackCheckpoint
from src.rag.document_manager import DocumentManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    target_channel = "SLACK_CHANNEL_ID_PLACEHOLDER"
    
    # 1. Init Slack Config & Source
    # Note: We hardcode parameters here to match presumed production config. 
    # Ideally we'd load from config/vatuta.yaml but this is a debug script.
    slack_config = SlackConfig(
        id="slack-main", 
        channel_types=["public_channel", "private_channel", "im", "mpim"], 
        workspace_domain="https://example.slack.com",
        initial_lookback_days=180,
        chunk_time_interval_minutes=240,
        chunk_max_size_chars=2000,
        chunk_max_count=20,
        chunk_similarity_threshold=0.15,
        chunk_embedding_model="all-MiniLM-L6-v2"
    )
    
    # Token from env
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        logger.error("SLACK_BOT_TOKEN env var is missing!")
        # Try to load from .env manually if needed, or assume user environment has it.
        # But 'poetry run' should handle it if dotenv plugin is active.
        return

    slack_source = SlackSource(
        config=slack_config,
        secrets={"bot_token": token},
        storage_path="data", # Sources usually append their source_id, e.g. data/slack
    )
    
    # Checkpoint
    checkpoint_path = Path("data") / "slack" / "slack-main" / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = SlackCheckpoint.load(checkpoint_path, slack_config)
    else:
        logger.info("No checkpoint found, creating new.")
        checkpoint = SlackCheckpoint(config=slack_config)

    # 2. Collect Chunks (Filtered)
    logger.info(f"Collecting data for channel {target_channel}...")
    # Using 'filters' argument which SlackSource supports
    docs, chunks = slack_source.collect_documents_and_chunks(
        checkpoint, 
        use_cached_data=True, 
        filters={"channel_ids": [target_channel]}
    )
    
    if not chunks:
        logger.warning(f"No chunks found for channel {target_channel}!")
        return

    logger.info(f"Found {len(docs)} documents and {len(chunks)} chunks.")
    
    # 3. Add to DocumentManager (VectorStore + Metadata)
    logger.info("Ingesting into DocumentManager...")
    # DocumentManager defaults to 'data' dir.
    dm = DocumentManager(storage_dir="data")
    
    # Build doc map
    doc_map = {d.document_id: d for d in docs}
    
    success = dm.add_chunk_records(chunks, doc_map)
    
    if success:
        logger.info("Successfully ingested channel specific data.")
    else:
        logger.error("Failed to ingest data.")

if __name__ == "__main__":
    main()
