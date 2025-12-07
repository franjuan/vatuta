"""Base configuration model for all data sources.

This module defines the common configuration fields shared by all sources.
"""

from pydantic import BaseModel, Field


class BaseSourceConfig(BaseModel):
    """Base configuration class for all data sources."""

    id: str = Field(..., description="Source ID")
    enabled: bool = Field(default=True, description="Whether this source is enabled")
