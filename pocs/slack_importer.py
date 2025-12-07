"""
Slack Importer Module

This module handles importing Slack channels and messages as LangChain Documents.
"""

import os
from typing import List, Optional

from langchain_core.documents import Document
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


class SlackImporter:
    """
    Handles importing Slack conversations into documents.
    """

    def __init__(self):
        self.slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
        self.slack_channel_types = os.getenv("SLACK_CHANNEL_TYPES", "public_channel,private_channel").strip()
        self.slack_oldest_ts = os.getenv("SLACK_OLDEST_TIMESTAMP")

        if not self.slack_bot_token:
            self.client = None
        else:
            self.client = WebClient(token=self.slack_bot_token)

    def import_slack(self, channel_name_filter: Optional[str] = None, limit_per_channel: int = 500) -> List[Document]:
        """
        Import Slack channels and messages as documents.

        Args:
            channel_name_filter: If provided, only channels whose name contains this substring
            limit_per_channel: Max messages to fetch per channel

        Returns:
            List[Document]
        """
        if not self.client:
            print("⚠️ Slack not configured. Set SLACK_BOT_TOKEN in environment.")
            return []

        types = self.slack_channel_types
        print(f"📨 Importing Slack conversations (types={types})")

        def list_conversations() -> list[dict]:
            conversations: list[dict] = []
            cursor = None
            while True:
                try:
                    resp = self.client.conversations_list(cursor=cursor, limit=200, types=types)
                    conversations.extend(resp.get("channels", []))
                    cursor = resp.get("response_metadata", {}).get("next_cursor")
                    if not cursor:
                        break
                except SlackApiError as e:
                    print(f"❌ Slack conversations_list error: {e.response.get('error')}")
                    break
            return conversations

        def list_messages(channel_id: str) -> list[dict]:
            messages: list[dict] = []
            cursor = None
            kwargs = {"channel": channel_id, "limit": 200}
            if self.slack_oldest_ts:
                kwargs["oldest"] = self.slack_oldest_ts
            while True:
                try:
                    resp = self.client.conversations_history(cursor=cursor, **kwargs)
                    messages.extend(resp.get("messages", []))
                    cursor = (
                        resp.get("response_metadata", {}).get("next_cursor") if "response_metadata" in resp else None
                    )
                    if not cursor or len(messages) >= limit_per_channel:
                        break
                except SlackApiError as e:
                    print(f"❌ Slack conversations_history error for {channel_id}: {e.response.get('error')}")
                    break
            return messages[:limit_per_channel]

        channels = list_conversations()
        if channel_name_filter:
            channels = [c for c in channels if channel_name_filter.lower() in c.get("name", "").lower()]

        docs: List[Document] = []
        for channel in channels:
            channel_id = channel.get("id")
            channel_name = channel.get("name", channel_id)
            messages = list_messages(channel_id)
            for msg in messages:
                text = msg.get("text", "")
                user = msg.get("user") or msg.get("username", "unknown")
                ts = msg.get("ts", "")
                content = f"Channel: {channel_name}\nUser: {user}\nTimestamp: {ts}\nMessage: {text}"
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": "slack",
                        "channel_id": channel_id,
                        "channel_name": channel_name,
                        "ts": ts,
                        "user": user,
                    },
                )
                docs.append(doc)

        print(f"✅ Imported {len(docs)} Slack messages from {len(channels)} channels")
        return docs


def main():
    importer = SlackImporter()
    importer.import_slack()


if __name__ == "__main__":
    main()
