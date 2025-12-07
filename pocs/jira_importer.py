"""
JIRA Ticket Importer Module

This module handles importing JIRA tickets and creating documents for the knowledge base.
"""

import os
from typing import List

from atlassian import Confluence
from jira import JIRA
from langchain_core.documents import Document
from langchain_text_splitters import SpacyTextSplitter


class JIRAImporter:
    """
    Handles importing JIRA tickets and Confluence pages into documents.
    """

    def __init__(self):
        """Initialize the JIRA importer with configuration from environment variables."""
        self.jira_user = os.getenv("JIRA_USER")
        self.jira_api_token = os.getenv("JIRA_API_TOKEN")
        self.jira_instance_url = os.getenv("JIRA_INSTANCE_URL")
        self.confluence_root = os.getenv("CONFLUENCE_ROOT")

        if not all([self.jira_user, self.jira_api_token, self.jira_instance_url]):
            raise ValueError("Missing required JIRA configuration. Please check your environment variables.")

        # Initialize JIRA connection
        self.jira_connection = JIRA(
            server=self.jira_instance_url,
            basic_auth=(self.jira_user, self.jira_api_token),
        )

        # Initialize Confluence connection
        self.confluence = Confluence(
            url=f"{self.jira_instance_url}/wiki",
            username=self.jira_user,
            password=self.jira_api_token,
        )

        # Initialize text splitter
        try:
            self.text_splitter = SpacyTextSplitter(pipeline="en_core_web_sm")
        except OSError:
            print(
                "Warning: spaCy model 'en_core_web_sm' not found. Please install it with: python -m spacy download en_core_web_sm"
            )
            self.text_splitter = None

    def import_jira_tickets(
        self,
        jql_query: str = "project = AL AND issuetype = Epic AND statusCategory != Done",
        max_results: int = 100,
    ) -> List[Document]:
        """
        Import JIRA tickets based on a JQL query.

        Args:
            jql_query: JQL query to search for tickets
            max_results: Maximum number of results to return

        Returns:
            List of Document objects containing ticket information
        """
        print(f"🔍 Searching JIRA with query: {jql_query}")

        try:
            issues = self.jira_connection.search_issues(jql_query, maxResults=max_results)
            print(f"📋 Found {len(issues)} tickets")

            docs = []
            for issue in issues:
                # Extract ticket information
                summary = issue.fields.summary
                description = issue.fields.description if issue.fields.description else ""
                status = issue.fields.status.name if issue.fields.status else "Unknown"
                assignee = issue.fields.assignee.displayName if issue.fields.assignee else "Unassigned"

                # Create content string
                content = f"""Issue Key: {issue.key}
Summary: {summary}
Status: {status}
Assignee: {assignee}
Description: {description}"""

                # Create document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": "jira",
                        "issue_id": issue.key,
                        "issue_type": "ticket",
                        "status": status,
                        "assignee": assignee,
                        "summary": summary,
                    },
                )
                docs.append(doc)
                print(f"✅ Imported ticket: {issue.key} - {summary}")

            return docs

        except Exception as e:
            print(f"❌ Error importing JIRA tickets: {e}")
            return []

    def import_confluence_pages(self, space_key: str = "RDT", limit: int = 100) -> List[Document]:
        """
        Import Confluence pages from a specific space.

        Args:
            space_key: Confluence space key
            limit: Maximum number of pages to import

        Returns:
            List of Document objects containing page information
        """
        if not self.confluence_root:
            print("⚠️ No Confluence root page ID configured. Skipping Confluence import.")
            return []

        print(f"📄 Importing Confluence pages from space: {space_key}")

        try:
            # Get all pages from the space
            pages = self.confluence.get_page_child_by_type(
                self.confluence_root,
                type="page",
                start=0,
                limit=limit,
                expand="body.storage",
            )

            print(f"📋 Found {len(pages)} Confluence pages")

            docs = []
            for page in pages:
                page_id = page["id"]
                title = page["title"]
                content = page["body"]["storage"]["value"]

                # Create document
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": "confluence",
                        "page_id": page_id,
                        "title": title,
                        "space_key": space_key,
                    },
                )
                docs.append(doc)
                print(f"✅ Imported page: {title}")

            return docs

        except Exception as e:
            print(f"❌ Error importing Confluence pages: {e}")
            return []

    def create_document_list(self, jira_docs: List[Document], confluence_docs: List[Document]) -> List[Document]:
        """
        Combine JIRA tickets and Confluence pages into a single document list.

        Args:
            jira_docs: List of JIRA ticket documents
            confluence_docs: List of Confluence page documents

        Returns:
            Combined list of all documents
        """
        all_docs = jira_docs + confluence_docs
        print(f"📚 Total documents created: {len(all_docs)}")
        print(f"   - JIRA tickets: {len(jira_docs)}")
        print(f"   - Confluence pages: {len(confluence_docs)}")

        return all_docs

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks using spaCy text splitter.

        Args:
            docs: List of documents to split

        Returns:
            List of split documents
        """
        if not self.text_splitter:
            print("⚠️ Text splitter not available. Returning original documents.")
            return docs

        print("✂️ Splitting documents into smaller chunks...")

        split_docs = []
        for doc in docs:
            try:
                split_docs.extend(self.text_splitter.split_documents([doc]))
            except Exception as e:
                print(
                    f"⚠️ Error splitting document {doc.metadata.get('issue_id', doc.metadata.get('page_id', 'unknown'))}: {e}"
                )
                # Keep original document if splitting fails
                split_docs.append(doc)

        print(f"📄 Split into {len(split_docs)} document chunks")
        return split_docs


def main():
    """
    Main function to demonstrate JIRA import functionality.
    """
    print("🚀 Starting JIRA Import Process")
    print("=" * 50)

    try:
        # Initialize importer
        importer = JIRAImporter()

        # Import JIRA tickets
        jira_docs = importer.import_jira_tickets()

        # Import Confluence pages
        confluence_docs = importer.import_confluence_pages()

        # Combine all documents
        all_docs = importer.create_document_list(jira_docs, confluence_docs)

        # Split documents if needed
        final_docs = importer.split_documents(all_docs)

        print("\n✅ Import process completed successfully!")
        print(f"📊 Final document count: {len(final_docs)}")

        return final_docs

    except Exception as e:
        print(f"❌ Import process failed: {e}")
        return []


if __name__ == "__main__":
    main()
