"""
FastAPI backend for Tab Clustering service.

Provides REST endpoints for clustering browser tabs using AI embeddings.

"""

import os
from typing import List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.cluster import cluster_tabs
from app.labeler import generate_cluster_name
from app.schemas import ClusterRequest, ClusterResponse, ClusterItem, Tab


# -----------------------------
# Initialize FastAPI App
# -----------------------------
app = FastAPI(
    title="AI Tab Clusterer",
    description="Cluster browser tabs into semantic groups using AI embeddings + offline TF-IDF labeler",
    version="3.0.0",
)


# -----------------------------
# CORS Configuration (Required for Chrome extension + Railway)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Global Exception Handler
# -----------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "path": request.url.path,
        },
    )


# -----------------------------
# Health Check
# -----------------------------
@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}


# -----------------------------
# Main Clustering Endpoint
# -----------------------------
@app.post("/cluster", response_model=ClusterResponse, tags=["clustering"])
async def cluster_tabs_endpoint(request: ClusterRequest) -> ClusterResponse:
    """
    Cluster the provided tabs into semantic groups and assign offline labels.
    """
    try:
        # Normalize input
        tabs_normalized = [
            {
                "title": tab.title.strip(),
                "url": str(tab.url).strip(),
            }
            for tab in request.tabs
        ]

        # Step 1 — Clustering (returns only id + tabs)
        raw_clusters = cluster_tabs(tabs_normalized)

        # Step 2 — Labeling (TF-IDF)
        cluster_items: List[ClusterItem] = []
        for c in raw_clusters:
            titles = [t["title"] for t in c["tabs"]]
            label = generate_cluster_name(titles)

            cluster_items.append(
                ClusterItem(
                    id=c["id"],
                    name=label,
                    tabs=[Tab(title=t["title"], url=t["url"]) for t in c["tabs"]],
                )
            )

        return ClusterResponse(clusters=cluster_items)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")


# -----------------------------
# Railway / Local Run
# -----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))  # Railway injects PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
