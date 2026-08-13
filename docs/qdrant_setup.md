# Qdrant Setup Guide

This guide details how to set up and manage the Qdrant vector database for Vatuta.

## Prerequisites

- **Docker**: Qdrant runs as a Docker container. Ensure Docker Desktop or Docker Engine is installed and running.
- **Just**: (Optional) The `justfile` includes shortcuts for managing Qdrant.

## Installation & Startup

We provide `just` commands to manage the Qdrant lifecycle easily.

1. **Start Qdrant**:

    ```bash
    just qdrant-start
    ```

    This pulls the `qdrant/qdrant:latest` image and starts a container named `vatuta-qdrant` on ports
    6333 (HTTP) and 6334 (GRPC). Data is persisted in `data/qdrant`.

2. **Check Status**:

    ```bash
    just qdrant-status
    ```

3. **View Logs**:

    ```bash
    just qdrant-logs
    ```

4. **Stop Qdrant**:

    ```bash
    just qdrant-stop
    ```

5. **Restart**:

    ```bash
    just qdrant-restart
    ```

## Configuration

### Environment Variables

Qdrant configuration is managed via `.env`. Ensure the following variable is set:

- `QDRANT_API_KEY`: A secure API key for authentication. You can generate a random string for this.

Example `.env`:

```bash
QDRANT_API_KEY=your_secure_random_key_here
```

### Application Config

The application connects to Qdrant using settings in `config/vatuta.yaml`. Vatuta supports **Hybrid Search**,
combining Semantic Search (dense vectors) with Lexical Search (BM25 sparse vectors).

```yaml
qdrant:
  url: "http://localhost:6333"
  collection_name: "vatuta_documents"

  # Semantic Search (Dense)
  embeddings_model: "intfloat/multilingual-e5-small"
  embeddings_query_prefix: "query: "       # Optional: prefix prepended to search queries
  embeddings_document_prefix: "passage: "   # Optional: prefix prepended to documents being indexed
  embeddings_normalize: true                # Optional: L2 normalize dense embeddings (defaults to false)
  # dense_vector_name: "dense" # Optional, defaults to "dense"

  # Lexical Search (Sparse / BM25)
  sparse_embeddings_model: "Qdrant/bm25"
  # sparse_vector_name: "sparse" # Optional, defaults to "sparse"
```

### Hybrid Search (BM25 + Semantic)

Vatuta automatically uses **RetrievalMode.HYBRID**. This means:

- **Semantic Search**: Driven by HuggingFace models (`embeddings_model`). Optional prefix fields
  (`embeddings_query_prefix` and `embeddings_document_prefix`) can be set in the configuration to prepend task
  instructions (e.g., `"query: "` and `"passage: "` for E5 models) to search queries and documents respectively.
- **Lexical Search (BM25)**: Evaluated locally via `fastembed` (`sparse_embeddings_model`) and matched natively by Qdrant.
- **Fusion**: Results are merged automatically using Qdrant's Reciprocal Rank Fusion (RRF).

## Dashboard

Qdrant comes with a built-in web UI dashboard.

- **URL**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
- **Launch Command**: `just qdrant-dashboard`

## Troubleshooting

### Container Name Conflict

**Error**: `Conflict. The container name "/vatuta-qdrant" is already in use`
**Fix**: Run `just qdrant-stop` to remove the old container before starting a new one.

### Authorization Error

**Error**: `Unexpected Response: 401 (Unauthorized)`
**Fix**: Ensure `QDRANT_API_KEY` is set in your `.env` file and that you have restarted the Qdrant container
(`just qdrant-restart`) after changing the key.

### Permission Denied on Storage

**Error**: `PermissionDenied ... path: "./storage/collections/vatuta_documents/0/wal/..."`
**Cause**: Files in `data/qdrant/` were previously created with `root` ownership (e.g., standard Qdrant
container image), so `qdrant/qdrant:latest-unprivileged` cannot write to them.
**Fix**: Fix file ownership with:

```bash
sudo chown -R $USER:$USER data/qdrant
```

Or reset the storage directory if data can be re-indexed:

```bash
sudo rm -rf data/qdrant
```

### Data Persistence

**Note**: Data is mounted to `$(pwd)/data/qdrant` (or `{{justfile_directory()}}/data/qdrant`).

To prevent Docker from creating the persistence directory as `root` or generating root-owned database files inside it,
we use the following setup:

1. **Pre-creation**: The `just qdrant-start` recipe pre-creates `data/qdrant` on the host using your local system
user.
2. **Unprivileged Image**: We run the official unprivileged image (`qdrant/qdrant:latest-unprivileged`). This image
runs Qdrant as a non-root user (UID `1000`) inside the container. Because of this, all generated database files
in your local `data/qdrant` folder are mapped directly to your local system user and not to `root`.
