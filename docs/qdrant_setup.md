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

The application connects to Qdrant using settings in `config/vatuta.yaml`:

```yaml
qdrant:
  url: "http://localhost:6333"
  collection_name: "vatuta_documents"
  embeddings_model: "sentence-transformers/all-MiniLM-L6-v2"
```

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

### Data Persistence

**Note**: Data is mounted to `$(pwd)/data/qdrant` (or `{{justfile_directory()}}/data/qdrant`).
Ensure this directory has write permissions.
