"""
Pydantic schemas for Tab Clustering API.

Defines request/response models for the /cluster endpoint.
"""

from typing import List
from pydantic import BaseModel, Field, HttpUrl, validator


class Tab(BaseModel):
    """Represents a single browser tab."""

    title: str = Field(..., min_length=1, max_length=300, description="Title of the tab")
    url: HttpUrl = Field(..., description="URL of the tab")

    @validator("title")
    def clean_title(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Tab title cannot be empty or whitespace")
        return v


class ClusterRequest(BaseModel):
    """Request payload for clustering tabs."""

    tabs: List[Tab] = Field(..., min_items=1, description="List of tabs to cluster")

    class Config:
        schema_extra = {
            "example": {
                "tabs": [
                    {"title": "GitHub - Project", "url": "https://github.com/user/project"},
                    {"title": "React docs", "url": "https://react.dev/learn"},
                    {"title": "Amazon", "url": "https://amazon.com"},
                    {"title": "YouTube", "url": "https://youtube.com"},
                    {"title": "Stack Overflow", "url": "https://stackoverflow.com"},
                ]
            }
        }


class ClusterItem(BaseModel):
    """Represents a single cluster with its tabs."""

    id: int = Field(..., ge=0, description="Cluster ID")
    name: str = Field(..., min_length=1, max_length=50, description="Name/category of the cluster")
    tabs: List[Tab] = Field(..., description="List of tabs in this cluster")


class ClusterResponse(BaseModel):
    """Response payload from the clustering endpoint."""

    clusters: List[ClusterItem] = Field(..., description="List of clusters")

    class Config:
        schema_extra = {
            "example": {
                "clusters": [
                    {
                        "id": 0,
                        "name": "Coding",
                        "tabs": [
                            {"title": "GitHub - Project", "url": "https://github.com/user/project"},
                            {"title": "Stack Overflow", "url": "https://stackoverflow.com"},
                        ],
                    },
                    {
                        "id": 1,
                        "name": "Shopping",
                        "tabs": [
                            {"title": "Amazon", "url": "https://amazon.com"},
                        ],
                    },
                ]
            }
        }
