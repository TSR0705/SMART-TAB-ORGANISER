"""
Advanced clustering pipeline for AI Tab Clusterer.

Pipeline:
1. Normalize titles & URLs
2. Create MiniLM embeddings (loaded once)
3. Run KMeans for a stable first-pass clustering
4. Identify small / noisy clusters and reassign using:
   - DBSCAN for local structure (optional)
   - nearest-cluster by cosine similarity (cheap & robust)
5. Merge near-duplicate clusters
6. Produce cluster dicts with id, tabs (no labels) — labels come from labeler.py
"""

from typing import List, Optional, Tuple, Dict
import re
from urllib.parse import urlparse, urlunparse
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load embedding model once
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Configuration
MAX_CLUSTERS = 8
MIN_CLUSTER_SIZE = 2      # clusters smaller than this are considered small/noise
REASSIGN_SIMILARITY_THRESH = 0.65  # threshold to glue small clusters to nearest
DBSCAN_EPS = 0.6          # eps for DBSCAN (on normalized embeddings)
DBSCAN_MIN_SAMPLES = 2


def normalize_url(url: str) -> str:
    """Normalize URL: remove query + fragment, lowercase, strip trailing slashes."""
    try:
        p = urlparse(url)
        # remove query and fragment
        clean = urlunparse((p.scheme, p.netloc, p.path.rstrip('/'), "", "", ""))
        return clean.lower()
    except Exception:
        return url.strip().lower()


def domain_from_url(url: str) -> str:
    try:
        p = urlparse(url)
        return p.netloc.lower()
    except Exception:
        return ""


def preprocess_tabs(tabs: List[dict]) -> List[dict]:
    """Return tabs with normalized titles/urls and helper fields."""
    normalized = []
    for t in tabs:
        title = (t.get("title") or "").strip()
        url = (t.get("url") or "").strip()
        url_norm = normalize_url(url)
        domain = domain_from_url(url_norm)
        normalized.append({
            "title": title,
            "url": url,
            "url_norm": url_norm,
            "domain": domain
        })
    return normalized


def _auto_n_clusters(n_tabs: int) -> int:
    if n_tabs <= 4:
        return 2
    return min(MAX_CLUSTERS, max(2, int(n_tabs ** 0.5)))


def _merge_similar_clusters(cluster_embeddings: List[np.ndarray], merge_thresh: float = 0.85) -> List[int]:
    """
    Simple greedy merge: if two cluster centroids cosine similarity > merge_thresh,
    merge them. Return mapping old_cluster_idx -> new_cluster_idx.
    """
    if len(cluster_embeddings) <= 1:
        return list(range(len(cluster_embeddings)))

    cents = np.vstack(cluster_embeddings)
    sim = cosine_similarity(cents, cents)
    n = len(cents)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= merge_thresh:
                union(i, j)

    mapping = {}
    new_idx = 0
    for i in range(n):
        r = find(i)
        if r not in mapping:
            mapping[r] = new_idx
            new_idx += 1
        mapping[i] = mapping[r]

    return mapping


def cluster_tabs(tabs: List[dict], n_clusters: Optional[int] = None) -> List[dict]:
    """
    Main entry: input list of {title,url}, returns clusters = [{id, tabs:[{title,url,...}]}].
    """
    if not tabs:
        raise ValueError("Tabs list cannot be empty")

    # Preprocess
    tabs_norm = preprocess_tabs(tabs)
    texts = [f"{t['title']} {t['url_norm']}" for t in tabs_norm]

    # Embeddings
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    # Decide cluster count
    if n_clusters is None:
        n_clusters = _auto_n_clusters(len(tabs_norm))

    # First-pass: KMeans (fast + stable)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Group tabs by cluster
    clusters: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(idx)

    # Compute cluster centroids
    centroids = []
    centroid_cluster_ids = []
    for c_id, idxs in clusters.items():
        cent = embeddings[idxs].mean(axis=0)
        centroids.append(cent)
        centroid_cluster_ids.append(c_id)

    # Merge very similar clusters (greedy)
    merge_map = _merge_similar_clusters(centroids, merge_thresh=0.90)
    # Build merged clusters mapping: new_id -> list of member original cluster ids
    merged = {}
    for orig_i, new_i in enumerate(merge_map):
        merged.setdefault(new_i, []).append(centroid_cluster_ids[orig_i])

    # Build new clusters dict (new numeric ids)
    new_clusters = {}
    for new_idx, orig_cluster_ids in merged.items():
        new_clusters[new_idx] = []
        for oc in orig_cluster_ids:
            new_clusters[new_idx].extend(clusters[oc])

    # Now handle small clusters / noise: reassign small clusters to nearest big cluster
    # Compute new centroids
    new_centroids = []
    for idxs in new_clusters.values():
        new_centroids.append(embeddings[idxs].mean(axis=0))
    new_centroids = np.vstack(new_centroids)
    # For small clusters (len < MIN_CLUSTER_SIZE), reassign members to nearest centroid if similar enough
    final_clusters = {i: list(idxs) for i, idxs in enumerate(new_clusters.values())}
    for cid, idxs in list(final_clusters.items()):
        if len(idxs) < MIN_CLUSTER_SIZE:
            # For each member, find nearest cluster centroid (excluding its own)
            for tidx in idxs[:]:
                vec = embeddings[tidx].reshape(1, -1)
                sims = cosine_similarity(vec, new_centroids)[0]
                # set its own cluster sim very low so it can move if needed
                sims[cid] = -1.0
                best_idx = int(np.argmax(sims))
                if sims[best_idx] >= REASSIGN_SIMILARITY_THRESH:
                    # move tidx to best_idx
                    final_clusters[cid].remove(tidx)
                    final_clusters[best_idx].append(tidx)
            # if cluster emptied, delete
            if not final_clusters[cid]:
                final_clusters.pop(cid, None)

    # If too many singleton/noisy tabs remain, run DBSCAN locally on their embeddings and reassign
    # collect leftover indices where their cluster size is 1
    singletons = [tidx for cid, idxs in final_clusters.items() for tidx in idxs if len(final_clusters[cid]) == 1]
    if len(singletons) >= 2:
        sub_emb = embeddings[singletons]
        try:
            db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine").fit(sub_emb)
            db_labels = db.labels_
            # cluster db labels into a new cluster grouping if any group found
            offset = max(final_clusters.keys()) + 1 if final_clusters else 0
            dbmap = {}
            for i, lab in enumerate(db_labels):
                if lab == -1:
                    continue
                dbmap.setdefault(lab, []).append(singletons[i])
            for lab, members in dbmap.items():
                final_clusters[offset] = members
                offset += 1
            # remove db-mapped members from their previous singleton cluster
            for tidx in [m for members in dbmap.values() for m in members]:
                for cid, idxs in list(final_clusters.items()):
                    if tidx in idxs and len(idxs) == 1:
                        final_clusters[cid].remove(tidx)
                        if not final_clusters[cid]:
                            final_clusters.pop(cid, None)
        except Exception:
            # DBSCAN is optional — if it fails, ignore
            pass

    # Final pass: create output (assign stable ids)
    output = []
    assigned = set()
    for new_id, idxs in sorted(final_clusters.items()):
        # filter out possible empty
        idxs = [i for i in idxs if i not in assigned]
        if not idxs:
            continue
        assigned.update(idxs)
        cluster_tabs = []
        for i in idxs:
            t = tabs_norm[i]
            cluster_tabs.append({"title": t["title"], "url": t["url"], "url_norm": t["url_norm"], "domain": t["domain"]})
        output.append({"id": int(new_id), "tabs": cluster_tabs})

    # Any unassigned tabs (rare) -> put into Misc cluster id 9999
    unassigned = [i for i in range(len(tabs_norm)) if i not in assigned]
    if unassigned:
        misc_tabs = []
        for i in unassigned:
            t = tabs_norm[i]
            misc_tabs.append({"title": t["title"], "url": t["url"], "url_norm": t["url_norm"], "domain": t["domain"]})
        output.append({"id": 9999, "tabs": misc_tabs})

    # Ensure deterministic ordering by id (misc last)
    output = sorted(output, key=lambda x: (x["id"] == 9999, x["id"]))

    return output
