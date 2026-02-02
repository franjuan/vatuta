"""Date and Time related tools."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TimeTool(BaseTool):
    """Tool to get the current time."""

    name: str = "get_current_time"
    description: str = (
        "Returns the current server time in ISO format. Use this to resolve relative dates like 'yesterday' or 'last week'."
    )
    handle_tool_error: bool = True

    def _run(self) -> str:
        return datetime.now(timezone.utc).isoformat()


class DateFilterSchema(BaseModel):
    """Schema for date filtering parameters."""

    start_date: Optional[str] = Field(None, description="Start date in ISO format (YYYY-MM-DD) or ISO datetime.")
    end_date: Optional[str] = Field(None, description="End date in ISO format (YYYY-MM-DD) or ISO datetime.")


class DateFilterTool(BaseTool):
    """Tool to create date filters for vector search."""

    name: str = "date_filter"
    description: str = (
        "Useful for creating date filters when the user query implies a time range. "
        "You must calculate specific start/end dates based on the current time before calling this tool."
    )
    args_schema: Type[BaseModel] = DateFilterSchema
    handle_tool_error: bool = True

    def _run(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """Convert explicit dates to Qdrant filter."""
        if not start_date and not end_date:
            return {}

        # Construct Qdrant Range Filter for 'source_updated_at'
        time_range = {}
        if start_date:
            time_range["gte"] = start_date
        if end_date:
            time_range["lte"] = end_date

        return {"must": [{"key": "metadata.source_updated_at", "range": time_range}]}


class DateAdderSchema(BaseModel):
    """Schema for date arithmetic."""

    base_date: Optional[str] = Field(
        None, description="Base date in ISO format (e.g. '2023-10-01T12:00:00'). If not provided, uses current time."
    )
    weeks: int = Field(0, description="Number of weeks to add (can be negative).")
    days: int = Field(0, description="Number of days to add (can be negative).")
    hours: int = Field(0, description="Number of hours to add (can be negative).")
    minutes: int = Field(0, description="Number of minutes to add (can be negative).")
    seconds: int = Field(0, description="Number of seconds to add (can be negative).")
    milliseconds: int = Field(0, description="Number of milliseconds to add (can be negative).")
    microseconds: int = Field(0, description="Number of microseconds to add (can be negative).")


class DateAdderTool(BaseTool):
    """Tool to perform date arithmetic (add/subtract time)."""

    name: str = "date_adder"
    description: str = "Useful for calculating new dates by adding/subtracting time (days, hours) to a base date."
    args_schema: Type[BaseModel] = DateAdderSchema
    handle_tool_error: bool = True

    def _run(
        self,
        base_date: Optional[str] = None,
        weeks: int = 0,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        milliseconds: int = 0,
        microseconds: int = 0,
    ) -> str:
        """Calculate new date."""
        if not base_date:
            dt = datetime.now(timezone.utc)
        else:
            try:
                # Parse ISO string
                dt = datetime.fromisoformat(base_date.replace("Z", "+00:00"))
            except ValueError:
                # Fallback for simple date
                try:
                    dt = datetime.strptime(base_date, "%Y-%m-%d")
                except ValueError:
                    return f"Error: Invalid date format '{base_date}'. Use ISO 8601."

        # Ensure dt is timezone aware if obtained from simple date string, assume UTC for consistency logic if needed
        # but datetime.strptime returns naive.
        # If naive, make it UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        delta = timedelta(
            weeks=weeks,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            milliseconds=milliseconds,
            microseconds=microseconds,
        )
        new_dt = dt + delta
        return new_dt.isoformat()
