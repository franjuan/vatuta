"""Tests for EntityManager."""

import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from src.entities.manager import EntityManager
from src.entities.models import SourceReference


@pytest.fixture
def temp_cache_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


def test_create_new_user(temp_cache_dir: Path) -> None:
    """Test creating a new user entity."""
    manager = EntityManager(storage_path=str(temp_cache_dir / "entities.json"))

    user = manager.get_or_create_user(
        source_type="slack",
        source_id="T123",
        source_user_id="U1",
        user_data={"real_name": "Alice", "email": "alice@example.com"},
    )

    assert user.global_id is not None
    assert "Alice" in user.names
    assert "alice@example.com" in user.emails
    assert SourceReference("slack", "T123", "U1") in user.source_refs


def test_retrieve_existing_user(temp_cache_dir: Path) -> None:
    """Test retrieving an existing user by reference."""
    manager = EntityManager(storage_path=str(temp_cache_dir / "entities.json"))

    user1 = manager.get_or_create_user(
        source_type="slack",
        source_id="T123",
        source_user_id="U1",
        user_data={"real_name": "Alice", "email": "alice@example.com"},
    )

    # Same source reference
    user2 = manager.get_or_create_user(
        source_type="slack",
        source_id="T123",
        source_user_id="U1",
        user_data={"real_name": "Alice Co.", "email": "alice@example.com"},
    )

    assert user1.global_id == user2.global_id
    # Check that name was updated/added
    assert "Alice Co." in user2.names


def test_link_users_by_email(temp_cache_dir: Path) -> None:
    """Test linking users from different sources by email."""
    manager = EntityManager(storage_path=str(temp_cache_dir / "entities.json"))

    # User from Slack
    user_slack = manager.get_or_create_user(
        source_type="slack",
        source_id="T123",
        source_user_id="U1",
        user_data={"real_name": "Alice", "email": "alice@example.com"},
    )

    # User from Jira with same email
    user_jira = manager.get_or_create_user(
        source_type="jira",
        source_id="J1",
        source_user_id="ACCT1",
        user_data={"displayName": "Alice J.", "email": "ALICE@Example.com"},  # Case insensitive check
    )

    assert user_slack.global_id == user_jira.global_id
    assert len(user_jira.source_refs) == 2
    assert SourceReference("slack", "T123", "U1") in user_jira.source_refs
    assert SourceReference("jira", "J1", "ACCT1") in user_jira.source_refs


def test_persistence(temp_cache_dir: Path) -> None:
    """Test saving and loading entities."""
    json_path = temp_cache_dir / "entities.json"
    manager = EntityManager(storage_path=str(json_path))

    manager.get_or_create_user(
        source_type="slack",
        source_id="T123",
        source_user_id="U1",
        user_data={"real_name": "Bob", "email": "bob@example.com"},
    )

    # Reload from disk
    manager2 = EntityManager(storage_path=str(json_path))
    user_bob = manager2.get_user_by_source_id("slack", "T123", "U1")

    assert user_bob is not None
    assert "bob@example.com" in user_bob.emails
    assert "Bob" in user_bob.names
