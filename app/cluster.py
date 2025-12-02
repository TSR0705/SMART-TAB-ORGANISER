"""
Tab clustering functionality for AI Tab Clusterer.

Improvements / fixes:
- _merge_similar_clusters now returns a list mapping index -> merged index
  (previously returned a dict and was iterated incorrectly).
- Defensive checks added around embeddings, centroids and DBSCAN.
- Ensured outputs have deterministic sequential IDs (misc remains 9999).
- Minor robustness improvements and clearer comments.
"""

from typing import List, Optional, Dict
from urllib.parse import urlparse, urlunparse
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load embedding model once (fast)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Configuration
MAX_CLUSTERS = 8
MIN_CLUSTER_SIZE = 2      # clusters smaller than this are considered small/noise
REASSIGN_SIMILARITY_THRESH = 0.65  # threshold to glue small clusters to nearest
DBSCAN_EPS = 0.6          # eps for DBSCAN (on cosine metric)
DBSCAN_MIN_SAMPLES = 2


def normalize_url(url: str) -> str:
    """Normalize URL by removing query parameters and fragments."""
    try:
        p = urlparse(url)
        # urlunparse expects (scheme, netloc, path, params, query, fragment)
        clean = urlunparse((p.scheme, p.netloc, p.path.rstrip('/'), "", "", ""))
        return clean.lower()
    except Exception:
        return (url or "").strip().lower()


def domain_from_url(url: str) -> str:
    try:
        p = urlparse(url)
        return (p.netloc or "").lower()
    except Exception:
        return ""


def preprocess_tabs(tabs: List[dict]) -> List[dict]:
    """Normalize tab data for processing."""
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
    Merge similar cluster centroids based on cosine similarity threshold.

    Returns:
        A list `mapping` of length N where mapping[i] = new_cluster_index for original centroid i.
    """
    n = len(cluster_embeddings)
    if n <= 1:
        return list(range(n))

    # Stack into (n, d)
    cents = np.vstack(cluster_embeddings)
    sim = cosine_similarity(cents, cents)

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
            # symmetric matrix
            if sim[i, j] >= merge_thresh:
                union(i, j)

    # Map each root to a compact new index
    mapping_root_to_new = {}
    mapping = [0] * n
    next_idx = 0
    for i in range(n):
        r = find(i)
        if r not in mapping_root_to_new:
            mapping_root_to_new[r] = next_idx
            next_idx += 1
        mapping[i] = mapping_root_to_new[r]

    return mapping


def cluster_tabs(tabs: List[dict], n_clusters: Optional[int] = None) -> List[dict]:
    """
    Main clustering function that processes tabs and returns clustered groups.
    Returns a list of dicts: {"id": int, "tabs": [ {title,url,url_norm,domain}, ... ] }
    """
    if not tabs:
        raise ValueError("Tabs list cannot be empty")

    # Preprocess tabs
    tabs_norm = preprocess_tabs(tabs)
    texts = [f"{t['title']} {t['url_norm']}" for t in tabs_norm]

    # Generate embeddings - defensive
    try:
        embeddings = embedder.encode(texts, convert_to_numpy=True)
    except TypeError:
        # Older/newer sentence-transformers variants might not support convert_to_numpy param
        embeddings = np.array(embedder.encode(texts))
    except Exception as e:
        raise RuntimeError(f"Embedding generation failed: {e}")

    # Ensure 2-D array (N, D)
    embeddings = np.asarray(embeddings)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.ndim != 2:
        raise RuntimeError("Unexpected embeddings shape, expected 2-D array")

    # Decide number of clusters
    if n_clusters is None:
        n_clusters = _auto_n_clusters(len(tabs_norm))

    # First-pass: KMeans clustering
    # n_init=10 for sklearn compatibility
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Group tab indices by KMeans cluster id
    clusters_by_label: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        clusters_by_label.setdefault(int(lab), []).append(idx)

    # Compute centroids for each KMeans cluster
    centroid_list = []
    centroid_cluster_ids = []
    for c_id, idxs in sorted(clusters_by_label.items()):
        if not idxs:
            continue
        cent = embeddings[idxs].mean(axis=0)
        centroid_list.append(cent)
        centroid_cluster_ids.append(c_id)

    # If no centroids (shouldn't happen), fallback
    if not centroid_list:
        # everything as misc
        return [{"id": 9999, "tabs": [{"title": t["title"], "url": t["url"], "url_norm": t["url_norm"], "domain": t["domain"]} for t in tabs_norm]}]

    # Merge highly similar centroids
    merge_map = _merge_similar_clusters(centroid_list, merge_thresh=0.90)  # mapping length == len(centroid_list)

    # Build mapping: merged_index -> list of original KMeans cluster ids
    merged_clusters_map: Dict[int, List[int]] = {}
    for orig_idx, merged_idx in enumerate(merge_map):
        kmeans_cluster_id = centroid_cluster_ids[orig_idx]
        merged_clusters_map.setdefault(merged_idx, []).append(kmeans_cluster_id)

    # Build new clusters (keys are new sequential indices)
    new_clusters: Dict[int, List[int]] = {}
    for new_idx, orig_cluster_ids in sorted(merged_clusters_map.items()):
        members: List[int] = []
        for oc in orig_cluster_ids:
            members.extend(clusters_by_label.get(oc, []))
        new_clusters[new_idx] = members

    # Compute centroids for new clusters
    new_centroids = []
    for idxs in new_clusters.values():
        if not idxs:
            # safety: skip empty
            continue
        new_centroids.append(embeddings[idxs].mean(axis=0))
    if not new_centroids:
        # fallback - put everything into misc
        return [{"id": 9999, "tabs": [{"title": t["title"], "url": t["url"], "url_norm": t["url_norm"], "domain": t["domain"]} for t in tabs_norm]}]
    new_centroids = np.vstack(new_centroids)

    # Initialize final_clusters as enumeration of new_clusters.values() so ids are sequential
    final_clusters: Dict[int, List[int]] = {i: list(idxs) for i, idxs in enumerate(new_clusters.values())}

    # Reassign members of very small clusters to nearest cluster when similarity high enough
    # We iterate over a static copy of items to avoid mutation during iteration issues.
    for cid, idxs in list(final_clusters.items()):
        if not idxs:
            # skip empties
            continue
        if len(idxs) < MIN_CLUSTER_SIZE:
            # try to reassign each member individually
            for tidx in idxs[:]:  # iterate on a copy
                vec = embeddings[tidx].reshape(1, -1)
                sims = cosine_similarity(vec, new_centroids)[0]
                # Mark current cluster's similarity very low so it doesn't pick itself
                if cid < len(sims):
                    sims[cid] = -1.0
                best_idx = int(np.argmax(sims))
                if sims[best_idx] >= REASSIGN_SIMILARITY_THRESH:
                    # move tidx to best cluster
                    try:
                        final_clusters[cid].remove(tidx)
                    except ValueError:
                        pass
                    final_clusters.setdefault(best_idx, []).append(tidx)
            # if cluster became empty, remove it
            if not final_clusters[cid]:
                final_clusters.pop(cid, None)

    # Collect singleton indices (tabs that are alone in their cluster) for optional DBSCAN grouping
    singletons = [tidx for cid, idxs in final_clusters.items() for tidx in idxs if len(final_clusters.get(cid, [])) == 1]
    if len(singletons) >= 2:
        try:
            sub_emb = embeddings[singletons]
            if sub_emb.shape[0] >= 2:
                db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine").fit(sub_emb)
                db_labels = db.labels_
                # Map DBSCAN labels to new clusters starting from next available id
                next_id = max(final_clusters.keys()) + 1 if final_clusters else 0
                dbmap: Dict[int, List[int]] = {}
                for i, lab in enumerate(db_labels):
                    if lab == -1:
                        continue
                    dbmap.setdefault(int(lab), []).append(singletons[i])
                # Add DBSCAN-found groups
                for lab, members in sorted(dbmap.items()):
                    final_clusters[next_id] = members
                    next_id += 1
                # Remove db-mapped members from their previous singleton clusters
                db_mapped = {m for members in dbmap.values() for m in members}
                for cid in list(final_clusters.keys()):
                    idxs = final_clusters.get(cid, [])
                    if not idxs:
                        continue
                    # If cluster contains a db-mapped member and is a singleton, remove it
                    if len(idxs) == 1 and idxs[0] in db_mapped:
                        final_clusters.pop(cid, None)
        except Exception:
            # DBSCAN is optional — withstand failures gracefully
            pass

    # Build final output with stable sequential IDs (0..N-1). Keep misc as 9999 if any leftovers.
    output = []
    assigned = set()
    # Sort final_clusters by numeric key for determinism
    for new_id in sorted(final_clusters.keys()):
        idxs = [i for i in final_clusters[new_id] if i not in assigned]
        if not idxs:
            continue
        assigned.update(idxs)
        cluster_tabs = []
        for i in idxs:
            t = tabs_norm[i]
            cluster_tabs.append({"title": t["title"], "url": t["url"], "url_norm": t["url_norm"], "domain": t["domain"]})
        output.append({"id": int(new_id), "tabs": cluster_tabs})

    # Any unassigned tabs -> put into Misc cluster id 9999
    unassigned = [i for i in range(len(tabs_norm)) if i not in assigned]
    if unassigned:
        misc_tabs = []
        for i in unassigned:
            t = tabs_norm[i]
            misc_tabs.append({"title": t["title"], "url": t["url"], "url_norm": t["url_norm"], "domain": t["domain"]})
        output.append({"id": 9999, "tabs": misc_tabs})

    # Deterministic ordering: non-misc clusters first (by id), misc last
    output = sorted(output, key=lambda x: (x["id"] == 9999, x["id"]))

    return output
