import gzip
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from src.models.documents import DocumentUnit
from src.sources.jira_source import JiraCheckpoint, JiraConfig, JiraSource


class TestJiraSource(unittest.TestCase):
    def setUp(self) -> None:
        # Use tempfile for test isolation
        self.temp_dir = tempfile.mkdtemp(prefix="test_jira_")
        self.config = {
            "url": "https://jira.example.com",
            "jql_query": "project = '{project}' AND updated >= '{updated_since}' order by updated ASC",
            "initial_lookback_days": 7,
            "include_comments": True,
            "projects": ["TEST", "DEV"],
            "id": "test_jira",
            "storage_path": self.temp_dir,
        }
        self.secrets = {"jira_user": "user", "jira_api_token": "token"}

        # Mock JIRA client
        self.mock_jira_patcher = patch("src.sources.jira_source.JIRA")
        self.mock_jira_cls = self.mock_jira_patcher.start()
        self.mock_jira = self.mock_jira_cls.return_value

        # Clean up storage path if exists
        storage_path = str(self.config["storage_path"])
        if Path(storage_path).exists():
            shutil.rmtree(storage_path)

    def tearDown(self) -> None:
        self.mock_jira_patcher.stop()
        storage_path = str(self.config["storage_path"])
        if Path(storage_path).exists():
            shutil.rmtree(storage_path)

    def test_collect_documents_and_chunks(self) -> None:
        # Use a recent date for updated so it's captured by the lookback window
        now = datetime.now(timezone.utc)
        updated_str = now.strftime("%Y-%m-%dT%H:%M:%S.%f%z")

        # Setup mock issue
        # To test history chunking/splitting, we need many HISTORY ENTRIES
        histories = []
        # We'll create 25 entries (default chunk size 20)
        for i in range(25):
            histories.append(
                {
                    "author": {"displayName": f"User{i}"},
                    "created": f"2023-10-28T10:{i:02d}:00.000+0000",
                    "items": [{"field": "status", "fromString": "Open", "toString": "In Progress"}],
                }
            )

        mock_issue = {
            "key": "TEST-1",
            "self": "https://jira.example.com/rest/api/2/issue/125142",
            "fields": {
                "summary": "Test Issue",
                "description": "Test Description",
                "status": {"name": "Open"},
                "priority": {"name": "High"},
                "issuetype": {"name": "Bug"},
                "project": {"key": "TEST"},
                "assignee": {"displayName": "John Doe"},
                "reporter": {"displayName": "Jane Doe"},
                "created": "2023-10-27T10:00:00.000+0000",
                "updated": updated_str,
                "labels": ["bug", "urgent"],
                "components": [{"name": "Backend"}],
                # Test taggeable custom field
                "customfield_10001": "MyCustomValue",
                # Test standard field in taggeable keys to verify deduplication
                # We'll put "priority" in taggeable_fields config below
                "parent": {"key": "TEST-PARENT"},
                "subtasks": [{"key": "TEST-SUB-1"}],
                "issuelinks": [],
                "comment": {"comments": []},
                # Empty field to verify inclusion
                "customfield_empty": None,
            },
            "changelog": {"histories": histories},
        }

        # Configure taggeable fields
        # "priority" is standard, "customfield_10001" is custom.
        self.config["taggeable_fields"] = ["customfield_10001", "priority"]
        self.config["history_chunk_size"] = 20
        # Use single project to avoid duplicate docs in test with same mock
        self.config["projects"] = ["TEST"]

        # enhanced_search_issues with json_result=True returns a dict
        self.mock_jira.enhanced_search_issues.return_value = {
            "issues": [mock_issue],
            "names": {
                "summary": "Summary",
                "description": "Description",
                "status": "Status",
                "assignee": "Assignee",
                "customfield_10001": "Custom Label",
                "customfield_empty": "Empty Field",
                "priority": "Priority",
            },
            "schema": {
                "customfield_10001": {"type": "string", "description": "My custom description"},
                "customfield_empty": {"type": "string"},
            },
        }

        # Initialize source
        source = JiraSource.create(config=JiraConfig(**self.config), data_dir=self.temp_dir, secrets=self.secrets)
        checkpoint = JiraCheckpoint(config=source.config)

        # Run collection
        docs, chunks = source.collect_documents_and_chunks(checkpoint)

        # Verify JQL
        self.assertEqual(self.mock_jira.enhanced_search_issues.call_count, 1)

        # Verify Document (1 doc)
        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertEqual(doc.document_id, "jira|test_jira|TEST-1")

        # Verify Tags
        # 1. Standard tags exist
        self.assertIn("status:Open", doc.system_tags)
        self.assertIn("priority:High", doc.system_tags)
        # 2. Custom tag exists
        self.assertIn("customfield_10001:MyCustomValue", doc.system_tags)
        # 3. Priority should NOT be duplicated (count should be 1 for 'priority:High')
        priority_tags = [t for t in doc.system_tags if t == "priority:High"]
        self.assertEqual(len(priority_tags), 1, "Standard field 'priority' should not be duplicated in tags")

        # Verify Chunks
        doc_chunks = [c for c in chunks if c.parent_document_id == doc.document_id]

        # 1. Ticket Body
        body_chunk = next(c for c in doc_chunks if "type:ticket" in c.system_tags)

        # Verify content formatting
        # - Key: Value
        # - Custom Label (My custom description): MyCustomValue
        # - Empty Field: None
        self.assertIn("Key: TEST-1", body_chunk.text)
        self.assertIn("Summary: Test Issue", body_chunk.text)
        self.assertIn("Custom Label (My custom description): MyCustomValue", body_chunk.text)
        self.assertIn("Empty Field: None", body_chunk.text)

        # 2. History Chunks (split)
        # We had 25 items, chunk size 20. Should be 2 chunks.
        hist_chunks = [c for c in doc_chunks if "type:history" in c.system_tags]
        self.assertEqual(len(hist_chunks), 2)

        # Verify first chunk has 20 items (approx, based on text parsing logic it's line based)
        # Each item is one line.
        chunk1_lines = hist_chunks[0].text.strip().split("\n")
        self.assertEqual(len(chunk1_lines), 20)

        # Verify second chunk has 5 items
        chunk2_lines = hist_chunks[1].text.strip().split("\n")
        self.assertEqual(len(chunk2_lines), 5)

        # Verify tags on hist chunks
        # Chunk 1 should have User0
        self.assertIn("author:User0", hist_chunks[0].system_tags)
        # Chunk 2 should have User20
        self.assertIn("author:User20", hist_chunks[1].system_tags)

        # Verify Checkpoint
        expected_ts = now.timestamp()
        self.assertAlmostEqual(checkpoint.get_project_updated_ts("TEST"), expected_ts, places=1)

    def test_ticket_content_formatting_sanitization(self) -> None:
        """Test that field values with newlines are sanitized into a single line."""
        issue = {
            "key": "TEST-2",
            "fields": {
                "summary": "Title\nWith\nNewlines",
                "description": "Line 1\r\nLine 2",
                "customfield_text": "Value with \r carriage return",
            },
        }
        names = {"customfield_text": "Custom Text", "summary": "Summary", "description": "Description"}
        schema: Any = {}

        # Instantiate source
        source = JiraSource(config=JiraConfig(**self.config), secrets=self.secrets, storage_path=self.temp_dir)

        # Call the private method
        content = source._format_ticket_content(issue, names, schema)

        # Check output
        lines = content.split("\n")

        # Parse lines to map Label -> Value
        data = {}
        for line in lines:
            if ": " in line:
                key, val = line.split(": ", 1)
                data[key] = val

        self.assertIn("Summary", data)
        self.assertIn("Description", data)
        self.assertIn("Custom Text", data)

        # Verify sanitization
        # "Title\nWith\nNewlines" -> "Title With Newlines"
        self.assertIn("Title With Newlines", data["Summary"])

        # "Line 1\r\nLine 2" -> "Line 1  Line 2" (assuming distinct replacement)
        # Note: replace("\r", " ").replace("\n", " ") converts \r\n to "  "
        self.assertIn("Line 1  Line 2", data["Description"])

        # "Value with \r carriage return" -> "Value with   carriage return"
        self.assertIn("Value with   carriage return", data["Custom Text"])

    @patch("src.sources.jira_source.gzip.open")
    def test_persistence(self, mock_gzip_open: MagicMock) -> None:
        # Enable caching
        self.config["use_cached_data"] = True

        # Setup mock issue
        mock_issue = {
            "key": "TEST-1",
            "fields": {
                "updated": "2023-10-27T10:00:00.000+0000",
                "summary": "Test Issue",
            },
        }

        self.mock_jira.enhanced_search_issues.return_value = {"issues": [mock_issue], "names": {}, "schema": {}}

        # Initialize source
        source = JiraSource.create(config=JiraConfig(**self.config), data_dir=self.temp_dir, secrets=self.secrets)
        checkpoint = JiraCheckpoint(config=source.config)

        # Mock file handle
        mock_file = MagicMock()
        mock_gzip_open.return_value.__enter__.return_value = mock_file

        # Run collection
        source.collect_documents_and_chunks(checkpoint)

        # Verify gzip.open was called for writing
        # Expected path: /tmp/test_jira_tests/test_jira/TEST/TEST.jsonl.gz
        # Note: source_id is "test_jira"

        # Check if gzip.open was called with the expected path and mode "wt"
        # We might have multiple calls (read and write), so we check if any call matches
        write_calls = [call for call in mock_gzip_open.mock_calls if "wt" in call.args or "wt" in call.kwargs.values()]
        self.assertTrue(len(write_calls) > 0, "gzip.open should be called with 'wt' mode")

        # Verify content written
        # We expect json.dumps(mock_issue) + "\n"
        # Since we mock existing issues as empty, it should just write the new issue
        mock_file.write.assert_called()
        written_args = [call.args[0] for call in mock_file.write.mock_calls]
        self.assertTrue(any("TEST-1" in arg for arg in written_args))

    def test_init_raises_on_auth_failure(self) -> None:
        # Mock myself() to raise an exception
        self.mock_jira.myself.side_effect = Exception("Auth failed")

        with self.assertRaises(ValueError) as cm:
            JiraSource.create(config=JiraConfig(**self.config), data_dir=self.temp_dir, secrets=self.secrets)

        self.assertIn("Failed to authenticate with JIRA", str(cm.exception))

    def test_collect_cached_documents_and_chunks(self) -> None:
        # Create real cache files
        # JiraSource.create uses data_dir/jira as storage_path, so files go in data_dir/jira/source_id/project
        base_dir = Path(self.temp_dir) / "jira" / str(self.config["id"])
        project_dir = base_dir / "TEST"
        project_dir.mkdir(parents=True, exist_ok=True)
        file_path = project_dir / "TEST.jsonl.gz"

        # Overlap logic test cases
        # Query Range: [2023-11-10, 2023-11-20]

        # 1. Issue completely before range: [2023-11-01, 2023-11-05] -> Exclude
        issue_before = {
            "key": "TEST-BEFORE",
            "fields": {
                "created": "2023-11-01T10:00:00.000+0000",
                "updated": "2023-11-05T10:00:00.000+0000",
                "summary": "Before",
            },
        }

        # 2. Issue completely after range: [2023-11-25, 2023-11-30] -> Exclude
        issue_after = {
            "key": "TEST-AFTER",
            "fields": {
                "created": "2023-11-25T10:00:00.000+0000",
                "updated": "2023-11-30T10:00:00.000+0000",
                "summary": "After",
            },
        }

        # 3. Issue overlaps start: [2023-11-05, 2023-11-15] -> Include
        issue_overlap_start = {
            "key": "TEST-OVERLAP-START",
            "fields": {
                "created": "2023-11-05T10:00:00.000+0000",
                "updated": "2023-11-15T10:00:00.000+0000",
                "summary": "Overlap Start",
            },
        }

        # 4. Issue overlaps end: [2023-11-15, 2023-11-25] -> Include
        issue_overlap_end = {
            "key": "TEST-OVERLAP-END",
            "fields": {
                "created": "2023-11-15T10:00:00.000+0000",
                "updated": "2023-11-25T10:00:00.000+0000",
                "summary": "Overlap End",
            },
        }

        # 5. Issue inside range: [2023-11-12, 2023-11-18] -> Include
        issue_inside = {
            "key": "TEST-INSIDE",
            "fields": {
                "created": "2023-11-12T10:00:00.000+0000",
                "updated": "2023-11-18T10:00:00.000+0000",
                "summary": "Inside",
            },
        }

        # 6. Range inside issue: [2023-11-01, 2023-11-30] -> Include
        issue_enclosing = {
            "key": "TEST-ENCLOSING",
            "fields": {
                "created": "2023-11-01T10:00:00.000+0000",
                "updated": "2023-11-30T10:00:00.000+0000",
                "summary": "Enclosing",
            },
        }

        with gzip.open(file_path, "wt", encoding="utf-8") as f:
            for issue in [
                issue_before,
                issue_after,
                issue_overlap_start,
                issue_overlap_end,
                issue_inside,
                issue_enclosing,
            ]:
                f.write(json.dumps(issue) + "\n")

        # Initialize source - use the same storage_path as configured in setUp
        source = JiraSource.create(
            config=JiraConfig(**self.config), data_dir=str(self.config["storage_path"]), secrets=self.secrets
        )

        # Test Date Filtering
        date_from = datetime(2023, 11, 10, tzinfo=timezone.utc)
        date_to = datetime(2023, 11, 20, tzinfo=timezone.utc)

        docs, chunks = source.collect_cached_documents_and_chunks(date_from=date_from, date_to=date_to)

        # Expected: Overlap Start, Overlap End, Inside, Enclosing (4 issues)
        # Excluded: Before, After

        doc_ids = [d.source_doc_id for d in docs]
        self.assertEqual(len(docs), 4)
        self.assertIn("TEST-OVERLAP-START", doc_ids)
        self.assertIn("TEST-OVERLAP-END", doc_ids)
        self.assertIn("TEST-INSIDE", doc_ids)
        self.assertIn("TEST-ENCLOSING", doc_ids)
        self.assertNotIn("TEST-BEFORE", doc_ids)
        self.assertNotIn("TEST-AFTER", doc_ids)

        # Test project_ids filtering
        docs, chunks = source.collect_cached_documents_and_chunks(filters={"project_ids": ["TEST"]})
        self.assertEqual(len(docs), 6)  # All issues

        docs, chunks = source.collect_cached_documents_and_chunks(filters={"project_ids": ["OTHER"]})
        self.assertEqual(len(docs), 0)

    @patch("src.sources.jira_source.JiraSource._get_embedding_model")
    def test_comment_chunking_strategies(self, mock_get_model: MagicMock) -> None:
        # Configuration for test
        self.config["chunk_max_count"] = 2
        self.config["chunk_max_size_chars"] = 1000
        self.config["chunk_similarity_threshold"] = 0.5  # High threshold to force split if low sim

        # Mock embedding model
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        # We define simple orthogonal vectors for controlling similarity
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]

        # Test 1: Count/Size Limit
        # We want high matching so no semantic split occurs.
        # Use vec_a for all.
        mock_model.encode.return_value = [vec_a, vec_a, vec_a, vec_a, vec_a]

        # Create comments
        # C1: len=5
        # C2: len=5. Total len 10. Count 2. -> Fits in Chunk1 (limit 1000 chars, 2 count).
        # C3: len=5. Count would be 3. -> SPLIT. Start Chunk2.
        # C4: len=15. Total len 20. Count 2. -> Fits in Chunk2.
        # C5: len=5. Count would be 3 -> SPLIT. Start Chunk3.

        comments = [
            {"author": {"displayName": "U1"}, "created": "2023-01-01T10:00", "body": "AAAAA"},  # 5 chars
            {"author": {"displayName": "U2"}, "created": "2023-01-01T11:00", "body": "BBBBB"},  # 5 chars
            {"author": {"displayName": "U3"}, "created": "2023-01-01T12:00", "body": "CCCCC"},  # 5 chars
            {"author": {"displayName": "U4"}, "created": "2023-01-01T13:00", "body": "DDDDDDDDDDDDDDD"},  # 15 chars
            {"author": {"displayName": "U5"}, "created": "2023-01-01T14:00", "body": "EEEEE"},  # 5 chars
        ]

        # Initialize source
        source = JiraSource(config=JiraConfig(**self.config), secrets=self.secrets, storage_path=self.temp_dir)
        doc = DocumentUnit(
            document_id="doc1",
            source="jira",
            source_doc_id="key1",
            source_instance_id="inst1",
            uri="uri",
            title="title",
            author="auth",
            parent_id=None,
            language=None,
            source_created_at=None,
            source_updated_at=None,
            system_tags=[],
            source_metadata={},
            content_hash="hash",
        )
        # Call _build_chunks_for_comments directly
        chunks = source._build_chunks_for_comments(doc, comments, ["tag:base"])

        # We expect 3 chunks based on count limit of 2.
        # Chunk 1: C1, C2
        # Chunk 2: C3, C4
        # Chunk 3: C5

        # Check lengths first
        self.assertEqual(len(chunks), 3)
        self.assertIn("AAAAA", chunks[0].text)
        self.assertIn("BBBBB", chunks[0].text)
        self.assertNotIn("CCCCC", chunks[0].text)

        self.assertIn("CCCCC", chunks[1].text)
        self.assertIn("DDDDDDDDDDDDDDD", chunks[1].text)

        self.assertIn("EEEEE", chunks[2].text)

        # Now test Size Limit
        # We want to test size limit.
        self.config["chunk_max_size_chars"] = 10000
        self.config["chunk_max_count"] = 2
        source = JiraSource(config=JiraConfig(**self.config), secrets=self.secrets, storage_path=self.temp_dir)
        chunks = source._build_chunks_for_comments(doc, comments, ["tag:base"])
        # Now Expect 3 chunks (2, 2, 1 items) because count limit is 2
        self.assertEqual(len(chunks), 3)

        # Test Semantic Splitting
        self.config["chunk_max_count"] = 10
        self.config["chunk_max_size_chars"] = 10000
        self.config["chunk_similarity_threshold"] = 0.8  # Split if < 0.8

        # C1, C2 -> High Sim (0.9) -> Join
        # C2, C3 -> Low Sim (0.1) -> Split
        # C3, C4 -> High Sim (0.9) -> Join
        # C4, C5 -> High Sim (0.9) -> Join

        # Embeddings:
        # E1: vec_a
        # E2: vec_a
        # E3: vec_b
        # E4: vec_b
        # E5: vec_b

        # Sim(E1, E2) = 1.0 > 0.8 (No split)
        # Sim(E2, E3) = 0.0 < 0.8 (SPLIT!)
        # Sim(E3, E4) = 1.0 > 0.8 (No split)
        # Sim(E4, E5) = 1.0 > 0.8 (No split)

        mock_model.encode.return_value = [vec_a, vec_a, vec_b, vec_b, vec_b]

        # Initialize new source with updated config
        source = JiraSource(config=JiraConfig(**self.config), secrets=self.secrets, storage_path=self.temp_dir)
        chunks = source._build_chunks_for_comments(doc, comments, ["tag:base"])

        # Expect: [C1, C2], [C3, C4, C5] -> 2 chunks.
        self.assertEqual(len(chunks), 2)
        self.assertIn("AAAAA", chunks[0].text)
        self.assertIn("BBBBB", chunks[0].text)
        self.assertNotIn("CCCCC", chunks[0].text)

        self.assertIn("CCCCC", chunks[1].text)
        self.assertIn("EEEEE", chunks[1].text)


if __name__ == "__main__":
    unittest.main()
