"""
Hybrid labeler for tab clustering.
"""

from typing import List, Dict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse

# -------------------------
# Category examples (expanded)

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
        "google docs", "documentation", "readthedocs", "developer docs", "manual"
    ],
    "Community": [
        "reddit thread", "discord chat", "forum discussion", "community post", "q&a"
    ],
    "Learning / Courses": [
        "udemy course", "course lesson", "lecture notes", "khanacademy", "edx course"
    ],
    "Search": [
        "search results", "query", "google search", "bing results"
    ],
    "Email / Communication": [
        "gmail inbox", "outlook mail", "send email", "mail client"
    ],
    "College Tools": [
        "srmist portal", "student dashboard", "seat finder", "exam seat", "college portal", "academic companion"
    ],
    "Math & CS Concepts": [
        "binary conversion", "decimal to binary", "algorithm explanation",
        "computer organisation", "restoring division", "non restoring division"
    ],
    "Music": [
        "lyrics", "song", "ghazal", "playlist", "music video"
    ],
    "Miscellaneous": [
        "misc", "other browsing", "random page", "unknown"
    ]
}

# -------------------------
# Domain -> Category quick map

# -------------------------
DOMAIN_CATEGORY_MAP = {
    "github.com": "Programming & Dev",
    "stackoverflow.com": "Programming & Dev",
    "youtube.com": "Video",
    "youtu.be": "Video",
    "amazon.com": "Shopping",
    "ebay.com": "Shopping",
    "medium.com": "Reading / Articles",
    "docs.google.com": "Docs",
    "notion.so": "Docs",
    "reddit.com": "Community",
    "discord.com": "Community",
    "udemy.com": "Learning / Courses",
    "coursera.org": "Learning / Courses",
    "khanacademy.org": "Learning / Courses",
    "srmist.edu.in": "College Tools",
    "srmit.edu.in": "College Tools",
}

# Keyword fallback map (expanded)
KEYWORD_MAP = {
    "Programming & Dev": ["github", "stack overflow", "stackoverflow", "gitlab", "code", "api", "npm", "python", "javascript"],
    "Video": ["youtube", "vimeo", "watch", "playlist", "stream"],
    "Shopping": ["amazon", "ebay", "flipkart", "buy", "product", "cart", "checkout"],
    "Reading / Articles": ["medium", "blog", "article", "news"],
    "Docs": ["docs.google", "notion", "readthedocs", "documentation", "manual"],
    "Community": ["reddit", "discord", "forum", "thread", "comments"],
    "Learning / Courses": ["udemy", "coursera", "khanacademy", "course", "lecture"],
    "Music": ["lyrics", "song", "ghazal", "music"],
    "Math & CS Concepts": ["binary", "decimal", "division", "algorithm", "computer organisation"],
    "College Tools": ["seat finder", "srm", "srmit", "campus", "exam seat", "helper"]
}

# -------------------------
# TF-IDF vectorizer built once
# -------------------------
_tfidf_vectorizer = None
_category_matrix = None
_category_names = None


def _build_tfidf():
    global _tfidf_vectorizer, _category_matrix, _category_names
    if _tfidf_vectorizer is not None:
        return

    docs = []
    names = []
    for cat, examples in CATEGORY_EXAMPLES.items():
        for ex in examples:
            docs.append(ex)
            names.append(cat)

    _tfidf_vectorizer = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
    _category_matrix = _tfidf_vectorizer.fit_transform(docs)
    _category_names = names


def _preprocess_titles(titles: List[str]) -> str:
    combined = " ".join([t or "" for t in titles]).lower()
    combined = re.sub(r"\s+", " ", combined).strip()
    return combined


def _domain_category(domain: str) -> Optional[str]:
    if not domain:
        return None
    domain = domain.lower()
    # try exact then suffix match
    if domain in DOMAIN_CATEGORY_MAP:
        return DOMAIN_CATEGORY_MAP[domain]
    for d, cat in DOMAIN_CATEGORY_MAP.items():
        if domain.endswith(d):
            return cat
    return None


def _keyword_fallback(text: str) -> str:
    text = text.lower()
    counts = {}
    for cat, keywords in KEYWORD_MAP.items():
        cnt = sum(1 for kw in keywords if kw in text)
        if cnt:
            counts[cat] = cnt
    if not counts:
        return "Miscellaneous"
    # return category with max matches
    return max(counts.items(), key=lambda x: x[1])[0]


def generate_cluster_name(titles: List[str], domains: List[str] = None) -> str:
    """
    Generate cluster name using multiple strategies.
    """
    _build_tfidf()
    titles_clean = _preprocess_titles(titles)
    # 1) domain unanimous mapping
    if domains:
        mapped = [d for d in ([_domain_category(d) for d in domains]) if d]
        if mapped:
            # if majority domain category, use it
            from collections import Counter
            c = Counter(mapped)
            cat, cnt = c.most_common(1)[0]
            if cnt >= 1 and len(mapped) / max(1, len(domains)) >= 0.5:
                return cat

    # 2) TF-IDF scoring
    try:
        v = _tfidf_vectorizer.transform([titles_clean])
        sims = cosine_similarity(v, _category_matrix)[0]
        # aggregate by category name (we stored one row per example)
        agg = {}
        for i, score in enumerate(sims):
            cat = _category_names[i]
            agg[cat] = max(agg.get(cat, 0.0), float(score))
        if agg:
            best_cat = max(agg.items(), key=lambda x: x[1])
            if best_cat[1] >= 0.12:   # threshold tuned for sensitivity
                return best_cat[0]
    except Exception:
        pass

    # 3) keyword fallback
    kw_cat = _keyword_fallback(titles_clean)
    if kw_cat:
        return kw_cat

    # 4) fallback
    return "Miscellaneous"
