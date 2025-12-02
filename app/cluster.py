"""
Tab clustering functionality for AI Tab Clusterer.

Improvements in this version:
- Fixed index mismatch bug in small cluster reassignment
- Fixed singleton detection logic to use snapshotted sizes
- Added validation for empty titles/URLs
- Improved embedding text construction with separator
- Extracted magic numbers to constants
- Enhanced type hints and docstrings
- Optimized centroid computation
- Better edge case handling
"""

from typing import List, Optional, Dict
from urllib.parse import urlparse, urlunparse
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load embedding model once (fast)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Configuration Constants
MAX_CLUSTERS = 8
MIN_CLUSTER_SIZE = 2  # clusters smaller than this are considered small/noise
REASSIGN_SIMILARITY_THRESH = 0.65  # threshold to glue small clusters to nearest
DBSCAN_EPS = 0.6  # eps for DBSCAN (on cosine metric)
DBSCAN_MIN_SAMPLES = 2
CENTROID_MERGE_THRESH = 0.90  # threshold for merging similar centroids
MISC_CLUSTER_ID = 9999  # ID for uncategorized tabs


def normalize_url(url: str) -> str:
    """
    Normalize URL by removing query parameters, fragments, and trailing slashes.
    
    Args:
        url: Raw URL string
        
    Returns:
        Normalized lowercase URL, or empty string on error
    """
    try:
        p = urlparse(url)
        # urlunparse expects (scheme, netloc, path, params, query, fragment)
        clean = urlunparse((p.scheme, p.netloc, p.path.rstrip('/'), "", "", ""))
        return clean.lower()
    except Exception:
        return (url or "").strip().lower()


def domain_from_url(url: str) -> str:
    """
    Extract domain from URL.
    
    Args:
        url: URL string
        
    Returns:
        Domain (netloc) in lowercase, or empty string on error
    """
    try:
        p = urlparse(url)
        return (p.netloc or "").lower()
    except Exception:
        return ""


def preprocess_tabs(tabs: List[dict]) -> List[dict]:
    """
    Normalize tab data for processing.
    
    Args:
        tabs: List of dicts with 'title' and 'url' keys
        
    Returns:
        List of normalized tab dicts with added 'url_norm' and 'domain' fields
    """
    normalized = []
    for t in tabs:
        # Validate and clean title/URL
        title = (t.get("title") or "").strip() or "Untitled"
        url = (t.get("url") or "").strip()
        
        # Skip tabs without valid URLs
        if not url or url.startswith("chrome://") or url.startswith("about:"):
            continue
            
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
    """
    Auto-determine optimal cluster count using logarithmic scaling.
    
    Args:
        n_tabs: Number of tabs to cluster
        
    Returns:
        Cluster count between 2 and MAX_CLUSTERS
    """
    if n_tabs <= 4:
        return 2
    # Log2 scaling provides better granularity than sqrt:
    # 8 tabs→3, 16→4, 32→5, 64→6, 128→7
    return min(MAX_CLUSTERS, max(2, int(np.log2(n_tabs))))


def _merge_similar_clusters(
    cluster_embeddings: List[np.ndarray], 
    merge_thresh: float = CENTROID_MERGE_THRESH
) -> List[int]:
    """
    Merge similar cluster centroids using Union-Find algorithm.

    Args:
        cluster_embeddings: List of centroid vectors
        merge_thresh: Cosine similarity threshold for merging (default 0.90)
        
    Returns:
        List mapping original centroid index → merged cluster index
    """
    n = len(cluster_embeddings)
    if n <= 1:
        return list(range(n))

    # Stack into (n, d) matrix
    centroids = np.vstack(cluster_embeddings)
    similarity_matrix = cosine_similarity(centroids, centroids)

    # Union-Find data structure
    parent = list(range(n))

    def find(x: int) -> int:
        """Find root with path compression."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # Path compression
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        """Union by rank (implicit via find order)."""
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    # Merge clusters above similarity threshold
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= merge_thresh:
                union(i, j)

    # Map each root to a compact sequential index
    root_to_new_id = {}
    mapping = [0] * n
    next_id = 0
    
    for i in range(n):
        root = find(i)
        if root not in root_to_new_id:
            root_to_new_id[root] = next_id
            next_id += 1
        mapping[i] = root_to_new_id[root]

    return mapping


def cluster_tabs(tabs: List[dict], n_clusters: Optional[int] = None) -> List[dict]:
    """
    Main clustering function using hybrid KMeans + DBSCAN approach.
    
    Pipeline:
    1. Preprocess and validate tabs
    2. Generate semantic embeddings
    3. Initial KMeans clustering
    4. Merge similar clusters
    5. Reassign small clusters to nearest neighbors
    6. Group remaining singletons with DBSCAN
    
    Args:
        tabs: List of dicts with 'title' and 'url' keys
        n_clusters: Optional fixed cluster count (auto-determined if None)
        
    Returns:
        List of cluster dicts: [{"id": int, "tabs": [...]}, ...]
        
    Raises:
        ValueError: If tabs list is empty
        RuntimeError: If embedding generation fails
    """
    if not tabs:
        raise ValueError("Tabs list cannot be empty")

    # Preprocess tabs (filters invalid URLs)
    tabs_norm = preprocess_tabs(tabs)
    
    if not tabs_norm:
        return [{"id": MISC_CLUSTER_ID, "tabs": []}]
    
    # Generate embedding texts with clear separator
    texts = [f"{t['title']} | {t['domain']}" for t in tabs_norm]

    # Generate embeddings with fallback handling
    try:
        embeddings = embedder.encode(texts, convert_to_numpy=True)
    except TypeError:
        # Older sentence-transformers versions don't support convert_to_numpy
        embeddings = np.array(embedder.encode(texts))
    except Exception as e:
        raise RuntimeError(f"Embedding generation failed: {e}")

    # Ensure 2-D array shape (N, D)
    embeddings = np.asarray(embeddings)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.ndim != 2:
        raise RuntimeError(f"Unexpected embeddings shape: {embeddings.shape}")

    # Auto-determine cluster count
    if n_clusters is None:
        n_clusters = _auto_n_clusters(len(tabs_norm))

    # Phase 1: KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Group tab indices by KMeans cluster ID
    clusters_by_label: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        clusters_by_label.setdefault(int(label), []).append(idx)

    # Compute centroids for each KMeans cluster
    centroid_list = []
    centroid_cluster_ids = []
    for cluster_id, tab_indices in sorted(clusters_by_label.items()):
        if not tab_indices:
            continue
        centroid = embeddings[tab_indices].mean(axis=0)
        centroid_list.append(centroid)
        centroid_cluster_ids.append(cluster_id)

    # Fallback if no valid clusters
    if not centroid_list:
        return [{
            "id": MISC_CLUSTER_ID, 
            "tabs": [{"title": t["title"], "url": t["url"], 
                     "url_norm": t["url_norm"], "domain": t["domain"]} 
                    for t in tabs_norm]
        }]

    # Phase 2: Merge highly similar clusters
    merge_mapping = _merge_similar_clusters(centroid_list, CENTROID_MERGE_THRESH)

    # Build mapping: merged_index -> list of original KMeans cluster IDs
    merged_clusters_map: Dict[int, List[int]] = {}
    for orig_idx, merged_idx in enumerate(merge_mapping):
        kmeans_cluster_id = centroid_cluster_ids[orig_idx]
        merged_clusters_map.setdefault(merged_idx, []).append(kmeans_cluster_id)

    # Consolidate tab indices into new merged clusters
    new_clusters: Dict[int, List[int]] = {}
    for new_idx, orig_cluster_ids in sorted(merged_clusters_map.items()):
        tab_indices: List[int] = []
        for orig_id in orig_cluster_ids:
            tab_indices.extend(clusters_by_label.get(orig_id, []))
        new_clusters[new_idx] = tab_indices

    # Compute centroids for merged clusters
    new_centroids = []
    for tab_indices in new_clusters.values():
        new_centroids.append(embeddings[tab_indices].mean(axis=0))
    
    if not new_centroids:
        return [{
            "id": MISC_CLUSTER_ID,
            "tabs": [{"title": t["title"], "url": t["url"],
                     "url_norm": t["url_norm"], "domain": t["domain"]}
                    for t in tabs_norm]
        }]
    
    new_centroids = np.array(new_centroids)

    # Initialize final clusters with sequential IDs
    final_clusters: Dict[int, List[int]] = {
        i: list(tab_indices) 
        for i, tab_indices in enumerate(new_clusters.values())
    }

    # Create mapping between cluster IDs and centroid indices
    sorted_cluster_ids = sorted(final_clusters.keys())
    cid_to_centroid_idx = {cid: idx for idx, cid in enumerate(sorted_cluster_ids)}

    # Phase 3: Reassign small clusters to nearest neighbors
    for cluster_id, tab_indices in list(final_clusters.items()):
        if not tab_indices or len(tab_indices) >= MIN_CLUSTER_SIZE:
            continue
        
        # Try reassigning each tab individually
        for tab_idx in tab_indices[:]:  # Iterate on copy
            vec = embeddings[tab_idx].reshape(1, -1)
            similarities = cosine_similarity(vec, new_centroids)[0]
            
            # Mask current cluster to avoid self-assignment
            if cluster_id in cid_to_centroid_idx:
                similarities[cid_to_centroid_idx[cluster_id]] = -1.0
            
            best_centroid_idx = int(np.argmax(similarities))
            best_similarity = similarities[best_centroid_idx]
            
            if best_similarity >= REASSIGN_SIMILARITY_THRESH:
                # Find cluster ID for best centroid
                best_cluster_id = sorted_cluster_ids[best_centroid_idx]
                
                # Move tab to best cluster
                try:
                    final_clusters[cluster_id].remove(tab_idx)
                except ValueError:
                    pass
                final_clusters.setdefault(best_cluster_id, []).append(tab_idx)
        
        # Remove empty clusters
        if not final_clusters[cluster_id]:
            final_clusters.pop(cluster_id, None)

    # Phase 4: Group remaining singletons with DBSCAN
    # Snapshot cluster sizes to avoid re-evaluation during iteration
    cluster_sizes = {cid: len(tab_indices) for cid, tab_indices in final_clusters.items()}
    singletons = [
        tab_idx 
        for cluster_id, tab_indices in final_clusters.items() 
        for tab_idx in tab_indices 
        if cluster_sizes[cluster_id] == 1
    ]
    
    if len(singletons) >= 2:
        try:
            singleton_embeddings = embeddings[singletons]
            if singleton_embeddings.shape[0] >= 2:
                db = DBSCAN(
                    eps=DBSCAN_EPS, 
                    min_samples=DBSCAN_MIN_SAMPLES, 
                    metric="cosine"
                ).fit(singleton_embeddings)
                
                db_labels = db.labels_
                
                # Map DBSCAN labels to new clusters
                next_id = max(final_clusters.keys()) + 1 if final_clusters else 0
                dbscan_clusters: Dict[int, List[int]] = {}
                
                for i, label in enumerate(db_labels):
                    if label == -1:  # Noise point
                        continue
                    dbscan_clusters.setdefault(int(label), []).append(singletons[i])
                
                # Add DBSCAN-found groups
                for label, members in sorted(dbscan_clusters.items()):
                    final_clusters[next_id] = members
                    next_id += 1
                
                # Remove old singleton clusters that were merged
                db_mapped_tabs = {
                    tab_idx 
                    for members in dbscan_clusters.values() 
                    for tab_idx in members
                }
                
                clusters_to_remove = [
                    cluster_id 
                    for cluster_id, tab_indices in final_clusters.items()
                    if len(tab_indices) == 1 and tab_indices[0] in db_mapped_tabs
                ]
                
                for cluster_id in clusters_to_remove:
                    final_clusters.pop(cluster_id, None)
                    
        except Exception as e:
            # DBSCAN is optional enhancement - don't fail entire clustering
            pass

    # Phase 5: Build final output with sequential IDs
    output = []
    assigned_tabs = set()
    
    # Process regular clusters
    for cluster_id in sorted(final_clusters.keys()):
        tab_indices = [i for i in final_clusters[cluster_id] if i not in assigned_tabs]
        if not tab_indices:
            continue
        
        assigned_tabs.update(tab_indices)
        cluster_tabs = []
        
        for i in tab_indices:
            t = tabs_norm[i]
            cluster_tabs.append({
                "title": t["title"], 
                "url": t["url"], 
                "url_norm": t["url_norm"], 
                "domain": t["domain"]
            })
        
        output.append({"id": int(cluster_id), "tabs": cluster_tabs})

    # Collect unassigned tabs into Miscellaneous cluster
    unassigned_indices = [
        i for i in range(len(tabs_norm)) 
        if i not in assigned_tabs
    ]
    
    if unassigned_indices:
        misc_tabs = []
        for i in unassigned_indices:
            t = tabs_norm[i]
            misc_tabs.append({
                "title": t["title"], 
                "url": t["url"], 
                "url_norm": t["url_norm"], 
                "domain": t["domain"]
            })
        output.append({"id": MISC_CLUSTER_ID, "tabs": misc_tabs})

    # Deterministic ordering: regular clusters first (by ID), misc last
    output = sorted(output, key=lambda x: (x["id"] == MISC_CLUSTER_ID, x["id"]))

    return output