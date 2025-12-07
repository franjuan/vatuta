import gzip
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.sources.jira import JiraCheckpoint, JiraConfig, JiraSource


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
        self.mock_jira_patcher = patch("src.sources.jira.JIRA")
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

        # Setup mock issue as dict (json_result=True format)
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
                "comment": {
                    "comments": [
                        {
                            "body": "First comment",
                            "author": {"displayName": "Alice"},
                            "created": "2023-10-27T11:00:00.000+0000",
                        }
                    ]
                },
            },
        }

        # enhanced_search_issues with json_result=True returns a dict
        self.mock_jira.enhanced_search_issues.return_value = {"issues": [mock_issue], "names": {}, "schema": {}}

        # Initialize source
        source = JiraSource.create(config=JiraConfig(**self.config), data_dir=self.temp_dir, secrets=self.secrets)
        checkpoint = JiraCheckpoint(config=source.config)

        # Run collection
        docs, chunks = source.collect_documents_and_chunks(checkpoint)

        # Verify JQL
        # Should be called twice (once for TEST, once for DEV)
        self.assertEqual(self.mock_jira.enhanced_search_issues.call_count, 2)

        # Check first call (TEST)
        call_args_1 = self.mock_jira.enhanced_search_issues.call_args_list[0]
        # enhanced_search_issues uses keyword arguments
        jql_arg_1 = (
            call_args_1.kwargs.get("jql")
            if call_args_1.kwargs
            else call_args_1[1].get("jql") if len(call_args_1) > 1 else None
        )
        if jql_arg_1:
            print(f"JQL 1 Used: {jql_arg_1}")
            self.assertIn("project = 'TEST'", jql_arg_1)
            self.assertIn("updated >=", jql_arg_1)

        # Verify Document
        # We mocked same issue for both calls, so we get 2 docs (duplicates in this mock scenario)
        self.assertEqual(len(docs), 2)
        doc = docs[0]
        self.assertEqual(doc.document_id, "jira|test_jira|TEST-1")
        self.assertEqual(doc.title, "[TEST-1] Test Issue")
        self.assertEqual(doc.source_metadata["status"], "Open")
        self.assertIn("status:Open", doc.system_tags)
        self.assertIn("label:urgent", doc.system_tags)
        self.assertIn("component:Backend", doc.system_tags)

        # Verify Chunks
        # 2 docs * 2 chunks each = 4 chunks
        self.assertEqual(len(chunks), 4)

        # Primary chunk
        primary = chunks[0]
        self.assertEqual(primary.chunk_index, 0)
        # Check for markdown format
        self.assertIn("# TEST-1: Test Issue", primary.text)
        self.assertIn("## Metadata", primary.text)
        self.assertIn("| **Status** | Open |", primary.text)
        self.assertIn("| **Labels** | bug, urgent |", primary.text)
        self.assertIn("## Description", primary.text)
        self.assertIn("type:description", primary.system_tags)

        # Comment chunk
        comment = chunks[1]
        self.assertEqual(comment.chunk_index, 1)
        self.assertIn("Comment by Alice", comment.text)
        self.assertIn("type:comment", comment.system_tags)

        # Verify Checkpoint
        # Should update to issue updated time for TEST project
        expected_ts = now.timestamp()
        self.assertAlmostEqual(checkpoint.get_project_updated_ts("TEST"), expected_ts, places=1)
        # DEV project also got updated because we returned same mock issue
        self.assertAlmostEqual(checkpoint.get_project_updated_ts("DEV"), expected_ts, places=1)

    @patch("src.sources.jira.gzip.open")
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


if __name__ == "__main__":
    unittest.main()
