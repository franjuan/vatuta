"""
GitLab Importer Module (PoC)

Fetch GitLab Issues and Merge Requests and convert them into LangChain
`Document` objects to be ingested by the local RAG vector store.

Environment variables:
- `GITLAB_URL`: Base URL of your GitLab instance (e.g., https://gitlab.com)
- `GITLAB_TOKEN`: Personal access token with API scope
- `GITLAB_PROJECT_IDS` (optional): Comma-separated numeric project IDs (e.g., "123,456")

Requirements:
- Dependency `python-gitlab` (declared in pyproject)

Quick start:
>>> from pocs.gitlab_importer import GitLabImporter
>>> importer = GitLabImporter()
>>> docs = importer.import_gitlab(include="both", limit_per_project=20)
>>> len(docs)

Each produced `Document` includes:
- page_content: A human-readable summary of the Issue/MR
- metadata: A dictionary with keys like `source`, `gitlab_project_id`,
  `gitlab_issue_iid` or `gitlab_mr_iid`, `title`, `state`, etc.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

import gitlab
from langchain_core.documents import Document


class GitLabImporter:
    """
    Importer for GitLab Issues and Merge Requests.

    Resolves project IDs from arguments or the `GITLAB_PROJECT_IDS` environment
    variable and exposes helpers to import issues and merge requests independently
    or together.
    """

    def __init__(self) -> None:
        """
        Initialize a GitLab API client using environment configuration.

        Reads:
        - `GITLAB_URL`
        - `GITLAB_TOKEN`
        - `GITLAB_PROJECT_IDS` (optional fallback for project selection)

        Raises:
            ValueError: If required environment variables are missing.
        """
        self.gitlab_url: Optional[str] = os.getenv("GITLAB_URL")
        self.gitlab_token: Optional[str] = os.getenv("GITLAB_TOKEN")
        self.env_project_ids = os.getenv("GITLAB_PROJECT_IDS", "").strip()

        if not self.gitlab_url or not self.gitlab_token:
            raise ValueError("Missing GitLab configuration. Set GITLAB_URL and GITLAB_TOKEN in environment.")

        self.client = gitlab.Gitlab(self.gitlab_url, private_token=self.gitlab_token)

    def _resolve_project_ids(self, project_ids: Optional[Iterable[int | str]]) -> List[int]:
        """
        Resolve project IDs from explicit arguments or environment.

        Args:
            project_ids: Iterable of numeric IDs or strings convertible to int.

        Returns:
            List[int]: Concrete list of project IDs.

        Raises:
            ValueError: If neither explicit `project_ids` nor `GITLAB_PROJECT_IDS`
                is provided.
        """
        if project_ids is not None:
            return [int(p) for p in project_ids]
        if self.env_project_ids:
            return [int(p.strip()) for p in self.env_project_ids.split(",") if p.strip()]
        raise ValueError("No project IDs provided. Pass --project-ids or set GITLAB_PROJECT_IDS.")

    def import_issues(
        self,
        project_ids: Optional[Iterable[int | str]] = None,
        state: Optional[str] = None,
        limit_per_project: int = 100,
    ) -> List[Document]:
        """
        Import issues from one or more GitLab projects.

        Args:
            project_ids: Iterable of project IDs; falls back to `GITLAB_PROJECT_IDS`.
            state: Optional state filter (e.g., "opened", "closed").
            limit_per_project: Safety limit to cap items per project.

        Returns:
            List[Document]: Documents representing the fetched issues.
        """
        docs: List[Document] = []
        for project_id in self._resolve_project_ids(project_ids):
            try:
                project = self.client.projects.get(project_id)
            except Exception as err:  # pragma: no cover - API errors are runtime dependent
                print(f"❌ Unable to access project {project_id}: {err}")
                continue

            try:
                issues = project.issues.list(state=state, per_page=min(100, limit_per_project), iterator=True)
            except Exception as err:  # pragma: no cover
                print(f"❌ Error listing issues for {project_id}: {err}")
                continue

            count = 0
            for issue in issues:
                if count >= limit_per_project:
                    break
                content = self._format_issue_content(project, issue)
                docs.append(
                    Document(
                        page_content=content,
                        metadata=self._issue_metadata(project, issue),
                    )
                )
                count += 1
        print(f"✅ Imported {len(docs)} GitLab issues")
        return docs

    def import_merge_requests(
        self,
        project_ids: Optional[Iterable[int | str]] = None,
        state: Optional[str] = None,
        limit_per_project: int = 100,
    ) -> List[Document]:
        """
        Import merge requests from one or more GitLab projects.

        Args:
            project_ids: Iterable of project IDs; falls back to `GITLAB_PROJECT_IDS`.
            state: Optional state filter (e.g., "opened", "closed", "merged").
            limit_per_project: Safety limit to cap items per project.

        Returns:
            List[Document]: Documents representing the fetched merge requests.
        """
        docs: List[Document] = []
        for project_id in self._resolve_project_ids(project_ids):
            try:
                project = self.client.projects.get(project_id)
            except Exception as err:  # pragma: no cover
                print(f"❌ Unable to access project {project_id}: {err}")
                continue

            try:
                mrs = project.mergerequests.list(state=state, per_page=min(100, limit_per_project), iterator=True)
            except Exception as err:  # pragma: no cover
                print(f"❌ Error listing merge requests for {project_id}: {err}")
                continue

            count = 0
            for mr in mrs:
                if count >= limit_per_project:
                    break
                content = self._format_mr_content(project, mr)
                docs.append(
                    Document(
                        page_content=content,
                        metadata=self._mr_metadata(project, mr),
                    )
                )
                count += 1
        print(f"✅ Imported {len(docs)} GitLab merge requests")
        return docs

    def import_gitlab(
        self,
        project_ids: Optional[Iterable[int | str]] = None,
        include: str = "both",
        state: Optional[str] = None,
        limit_per_project: int = 100,
    ) -> List[Document]:
        """
        High-level import that fetches issues and/or merge requests.

        Args:
            project_ids: Iterable of project IDs; falls back to `GITLAB_PROJECT_IDS`.
            include: What to import: "issues", "mrs"/"merge_requests", or "both".
            state: Optional state filter passed through to underlying calls.
            limit_per_project: Max items per project per type.

        Returns:
            List[Document]: Combined list of Documents from the selected types.
        """
        include = include.lower()
        docs: List[Document] = []
        if include in ("issues", "both"):
            docs.extend(
                self.import_issues(
                    project_ids=project_ids,
                    state=state,
                    limit_per_project=limit_per_project,
                )
            )
        if include in ("mrs", "merge_requests", "both"):
            docs.extend(
                self.import_merge_requests(
                    project_ids=project_ids,
                    state=state,
                    limit_per_project=limit_per_project,
                )
            )
        return docs

    def _format_issue_content(self, project: Any, issue: Any) -> str:
        """
        Create a concise, readable text representation of an issue for storage
        as `Document.page_content`.

        Includes: project path, issue IID, title, state, author, assignee,
        labels and description.
        """
        description = issue.attributes.get("description", "") or ""
        labels = ", ".join(issue.attributes.get("labels", []) or [])
        assignee = (
            (issue.attributes.get("assignee") or {}).get("name")
            if isinstance(issue.attributes.get("assignee"), dict)
            else None
        )
        author = (
            (issue.attributes.get("author") or {}).get("name")
            if isinstance(issue.attributes.get("author"), dict)
            else None
        )
        return (
            f"Project: {project.path_with_namespace}\n"
            f"Issue IID: {issue.iid}\n"
            f"Title: {issue.title}\n"
            f"State: {issue.state}\n"
            f"Author: {author or 'unknown'}\n"
            f"Assignee: {assignee or 'unassigned'}\n"
            f"Labels: {labels}\n"
            f"Description: {description}"
        )

    def _issue_metadata(self, project: Any, issue: Any) -> Dict[str, Any]:
        """
        Build metadata for an issue `Document`.

        Keys include:
        - `source` = "gitlab"
        - `gitlab_project_id`, `gitlab_project_path`
        - `gitlab_issue_iid`
        - `title`, `state`, `labels`, `author`, `assignee`
        - `item_type` = "issue"
        """
        return {
            "source": "gitlab",
            "gitlab_project_id": project.id,
            "gitlab_project_path": project.path_with_namespace,
            "gitlab_issue_iid": issue.iid,
            "title": issue.title,
            "state": issue.state,
            "labels": issue.attributes.get("labels", []) or [],
            "author": (issue.attributes.get("author") or {}).get("name"),
            "assignee": (
                (issue.attributes.get("assignee") or {}).get("name")
                if isinstance(issue.attributes.get("assignee"), dict)
                else None
            ),
            "item_type": "issue",
        }

    def _format_mr_content(self, project: Any, mr: Any) -> str:
        """
        Create a concise, readable text representation of a merge request for
        storage as `Document.page_content`.

        Includes: project path, MR IID, title, state, author, assignee,
        labels and description.
        """
        description = mr.attributes.get("description", "") or ""
        labels = ", ".join(mr.attributes.get("labels", []) or [])
        assignee = (
            (mr.attributes.get("assignee") or {}).get("name")
            if isinstance(mr.attributes.get("assignee"), dict)
            else None
        )
        author = (
            (mr.attributes.get("author") or {}).get("name") if isinstance(mr.attributes.get("author"), dict) else None
        )
        return (
            f"Project: {project.path_with_namespace}\n"
            f"MR IID: {mr.iid}\n"
            f"Title: {mr.title}\n"
            f"State: {mr.state}\n"
            f"Author: {author or 'unknown'}\n"
            f"Assignee: {assignee or 'unassigned'}\n"
            f"Labels: {labels}\n"
            f"Description: {description}"
        )

    def _mr_metadata(self, project: Any, mr: Any) -> Dict[str, Any]:
        """
        Build metadata for a merge request `Document`.

        Keys include:
        - `source` = "gitlab"
        - `gitlab_project_id`, `gitlab_project_path`
        - `gitlab_mr_iid`
        - `title`, `state`, `labels`, `author`, `assignee`
        - `item_type` = "merge_request"
        """
        return {
            "source": "gitlab",
            "gitlab_project_id": project.id,
            "gitlab_project_path": project.path_with_namespace,
            "gitlab_mr_iid": mr.iid,
            "title": mr.title,
            "state": mr.state,
            "labels": mr.attributes.get("labels", []) or [],
            "author": (mr.attributes.get("author") or {}).get("name"),
            "assignee": (
                (mr.attributes.get("assignee") or {}).get("name")
                if isinstance(mr.attributes.get("assignee"), dict)
                else None
            ),
            "item_type": "merge_request",
        }


def main() -> None:  # pragma: no cover - demo entry point
    """Demonstration entry point for manual testing of the importer."""
    print("🚀 GitLab Importer PoC")
    try:
        importer = GitLabImporter()
        docs = importer.import_gitlab(include="both", limit_per_project=20)
        print(f"📊 Imported {len(docs)} documents from GitLab")
    except Exception as e:
        print(f"❌ GitLab import failed: {e}")


if __name__ == "__main__":  # pragma: no cover
    main()
