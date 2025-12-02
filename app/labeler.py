"""
Improved, bug-fixed labeler for AI Tab Clusterer.
- Stronger TF-IDF scoring
- Smarter domain majority detection
- Safer keyword fallback
- Deterministic cluster labels
"""

from typing import List, Dict, Optional
import re
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# CATEGORY DEFINITIONS
# -------------------------

CATEGORY_EXAMPLES: Dict[str, List[str]] = {
    "Programming & Dev": [
        "github repository", "stack overflow question", "javascript tutorial",
        "python pip", "api reference", "developer docs", "npm package", "code snippet"
    ],
    "Video": [
        "youtube video", "watch video", "playlist", "video tutorial", "live stream"
    ],
    "Shopping": [
        "product listing", "shopping cart", "checkout", "product page", "buy now"
    ],
    "Reading / Articles": [
        "medium article", "blog post", "news article", "long read", "opinion piece"
    ],
    "Docs": [
        "google docs", "documentation", "readthedocs", "manual", "tech docs"
    ],
    "Community": [
        "reddit thread", "discord chat", "forum discussion", "community post"
    ],
    "Learning / Courses": [
        "udemy course", "course lesson", "lecture notes", "khanacademy", "edx course"
    ],
    "Search": [
        "search results", "query", "google search", "bing results"
    ],
    "Email / Communication": [
        "gmail inbox", "outlook mail", "send email"
    ],
    "College Tools": [
        "srmist portal", "student dashboard", "exam seat", "college portal"
    ],
    "Math & CS Concepts": [
        "binary conversion", "decimal to binary", "algorithm explanation",
        "computer organisation", "restoring division", "non restoring division"
    ],
    "Music": [
        "lyrics", "song", "playlist", "music video"
    ],
    "Miscellaneous": ["misc page", "random", "other"]
}

DOMAIN_CATEGORY_MAP = {
    "github.com": "Programming & Dev",
    "stackoverflow.com": "Programming & Dev",
    "youtube.com": "Video",
    "youtu.be": "Video",
    "amazon.com": "Shopping",
    "flipkart.com": "Shopping",
    "medium.com": "Reading / Articles",
    "docs.google.com": "Docs",
    "notion.so": "Docs",
    "reddit.com": "Community",
    "discord.com": "Community",
    "udemy.com": "Learning / Courses",
    "coursera.org": "Learning / Courses",
    "khanacademy.org": "Learning / Courses",
    "srmist.edu.in": "College Tools",
}

KEYWORD_MAP = {
    "Programming & Dev": ["github", "stack overflow", "gitlab", "code ", "api ", "npm", "python", "javascript"],
    "Video": ["youtube", "video", "playlist", "stream"],
    "Shopping": ["amazon", "flipkart", "buy", "product", "cart", "checkout"],
    "Reading / Articles": ["medium", "blog", "article", "news"],
    "Docs": ["docs.google", "notion", "readthedocs", "documentation"],
    "Community": ["reddit", "discord", "forum"],
    "Learning / Courses": ["course", "lecture", "udemy", "coursera", "khanacademy"],
    "Music": ["lyrics", "song", "music"],
    "Math & CS Concepts": ["binary", "algorithm", "division"],
    "College Tools": ["srm", "college portal", "exam seat"]
}

# -------------------------
# TF-IDF initialization
# -------------------------

_tfidf_vectorizer = None
_category_matrix = None
_category_names = None


def _build_tfidf():
    """Build TF-IDF matrix only once."""
    global _tfidf_vectorizer, _category_matrix, _category_names
    if _tfidf_vectorizer is not None:
        return

    docs = []
    names = []

    for cat, examples in CATEGORY_EXAMPLES.items():
        for ex in examples:
            docs.append(ex)
            names.append(cat)

    _tfidf_vectorizer = TfidfVectorizer(
        max_features=1800,
        stop_words="english",
        ngram_range=(1, 2)
    )
    _category_matrix = _tfidf_vectorizer.fit_transform(docs)
    _category_names = names


# -------------------------
# Helpers
# -------------------------

def _preprocess_titles(titles: List[str]) -> str:
    text = " ".join(set([t.lower().strip() for t in titles if t]))
    return re.sub(r"\s+", " ", text).strip()


def _domain_category(domain: str) -> Optional[str]:
    if not domain:
        return None
    domain = domain.lower()

    # exact match
    if domain in DOMAIN_CATEGORY_MAP:
        return DOMAIN_CATEGORY_MAP[domain]

    # suffix match
    for d, cat in DOMAIN_CATEGORY_MAP.items():
        if domain.endswith(d):
            return cat

    return None


def _keyword_fallback(text: str) -> str:
    text = text.lower()
    scores = {}

    for cat, keywords in KEYWORD_MAP.items():
        cnt = sum(1 for kw in keywords if kw in text)
        if cnt:
            scores[cat] = cnt

    return max(scores.items(), key=lambda x: x[1])[0] if scores else "Miscellaneous"


# -------------------------
# Main label generator
# -------------------------

def generate_cluster_name(titles: List[str], domains: List[str] = None) -> str:
    _build_tfidf()

    clean_titles = _preprocess_titles(titles)

    # -------------------------
    # 1. DOMAIN MAJORITY (strongest signal)
    # -------------------------
    if domains:
        mapped = [c for c in [_domain_category(d) for d in domains] if c]

        if mapped:
            from collections import Counter
            c = Counter(mapped)

            cat, cnt = c.most_common(1)[0]

            # Require >60% dominance to avoid accidental mislabel
            if cnt / len(domains) >= 0.60:
                return cat

    # -------------------------
    # 2. TF-IDF SCORING (primary)
    # -------------------------
    try:
        vec = _tfidf_vectorizer.transform([clean_titles])
        sims = cosine_similarity(vec, _category_matrix)[0]

        agg = {}
        for i, score in enumerate(sims):
            cat = _category_names[i]
            agg.setdefault(cat, []).append(float(score))

        # use average similarity per category for stability
        cat_scores = {cat: sum(vals)/len(vals) for cat, vals in agg.items()}

        best_cat, best_score = max(cat_scores.items(), key=lambda x: x[1])

        if best_score >= 0.10:
            return best_cat

    except Exception:
        pass

    # -------------------------
    # 3. KEYWORD FALLBACK
    # -------------------------
    kw_cat = _keyword_fallback(clean_titles)
    if kw_cat:
        return kw_cat

    # -------------------------
    # 4. DEFAULT
    # -------------------------
    return "Miscellaneous"
