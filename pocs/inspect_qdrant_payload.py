
from src.models.config import ConfigLoader
from src.rag.qdrant_manager import QdrantDocumentManager
import json

def main():
    config = ConfigLoader.load("config/vatuta.yaml")
    dm = QdrantDocumentManager(config.qdrant)
    
    # Scroll one point
    scroll_result, _ = dm.client.scroll(
        collection_name=dm.collection_name,
        limit=1,
        with_payload=True
    )
    
    if scroll_result:
        print("Payload sample:")
        print(json.dumps(scroll_result[0].payload, indent=2, default=str))
    else:
        print("No documents found.")

if __name__ == "__main__":
    main()
