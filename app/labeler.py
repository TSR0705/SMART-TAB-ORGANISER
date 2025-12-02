"""
Cluster labeling system for AI Tab Clusterer.

Hybrid approach using:
1. Domain-based classification (highest confidence)
2. TF-IDF semantic similarity (primary)
3. Keyword matching (fallback)

Improvements in this version:
- Fixed missing Counter import
- Added division-by-zero protection
- Improved keyword matching with word boundaries
- Preserved title frequency in preprocessing
- Added error handling for malformed inputs
- Extracted magic numbers to constants
- Enhanced category coverage (added News)
- Optimized scoring aggregation
"""

from typing import List, Dict, Optional
from collections import Counter, defaultdict
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# CONFIGURATION CONSTANTS
# -------------------------

TFIDF_MAX_FEATURES = 1800
DOMAIN_MAJORITY_THRESHOLD = 0.50  # 50% of tabs must share domain
TFIDF_SCORE_THRESHOLD = 0.12  # Minimum similarity score
NGRAM_RANGE = (1, 2)  # Unigrams and bigrams

# -------------------------
# CATEGORY DEFINITIONS
# -------------------------

CATEGORY_EXAMPLES: Dict[str, List[str]] = {
    "Programming & Dev": [
        "github repository", "stack overflow question", "javascript tutorial",
        "python pip", "api reference", "developer docs", "npm package", 
        "code snippet", "pull request", "issue tracker"
    ],
    "Video": [
        "youtube video", "watch video", "playlist", "video tutorial", 
        "live stream", "vimeo", "video content"
    ],
    "Shopping": [
        "product listing", "shopping cart", "checkout", "product page", 
        "buy now", "add to cart", "online store"
    ],
    "Reading / Articles": [
        "medium article", "blog post", "news article", "long read", 
        "opinion piece", "essay", "story"
    ],
    "Docs": [
        "google docs", "documentation", "readthedocs", "manual", 
        "tech docs", "api docs", "reference guide"
    ],
    "Community": [
        "reddit thread", "discord chat", "forum discussion", 
        "community post", "q&a", "discussion board"
    ],
    "Learning / Courses": [
        "udemy course", "course lesson", "lecture notes", "khanacademy", 
        "edx course", "tutorial", "learning path"
    ],
    "Search": [
        "search results", "query", "google search", "bing results", 
        "web search"
    ],
    "Email / Communication": [
        "gmail inbox", "outlook mail", "send email", "compose message",
        "mail client"
    ],
    "News": [
        "breaking news", "latest news", "news article", "journalism",
        "news report", "current events"
    ],
    "College Tools": [
        "srmist portal", "student dashboard", "exam seat", "college portal",
        "academic system", "university"
    ],
    "Math & CS Concepts": [
        "binary conversion", "decimal to binary", "algorithm explanation",
        "computer organisation", "restoring division", "non restoring division",
        "data structures"
    ],
    "Music": [
        "lyrics", "song", "playlist", "music video", "album", 
        "artist page", "spotify"
    ],
    "Social Media": [
        "twitter feed", "facebook post", "instagram", "social network",
        "profile page", "timeline"
    ],
    "Miscellaneous": [
        "misc page", "random", "other", "unknown"
    ]
}

# Domain to category mapping (highest confidence signal)
DOMAIN_CATEGORY_MAP = {
    # Development
    "github.com": "Programming & Dev",
    "gitlab.com": "Programming & Dev",
    "stackoverflow.com": "Programming & Dev",
    "stackexchange.com": "Programming & Dev",
    
    # Video
    "youtube.com": "Video",
    "youtu.be": "Video",
    "vimeo.com": "Video",
    
    # Shopping
    "amazon.com": "Shopping",
    "amazon.in": "Shopping",
    "flipkart.com": "Shopping",
    "ebay.com": "Shopping",
    
    # Reading
    "medium.com": "Reading / Articles",
    "substack.com": "Reading / Articles",
    
    # Docs
    "docs.google.com": "Docs",
    "notion.so": "Docs",
    "confluence.atlassian.com": "Docs",
    
    # Community
    "reddit.com": "Community",
    "discord.com": "Community",
    
    # Learning
    "udemy.com": "Learning / Courses",
    "coursera.org": "Learning / Courses",
    "khanacademy.org": "Learning / Courses",
    "edx.org": "Learning / Courses",
    
    # News
    "cnn.com": "News",
    "bbc.com": "News",
    "reuters.com": "News",
    "nytimes.com": "News",
    "theguardian.com": "News",
    
    # Social
    "twitter.com": "Social Media",
    "x.com": "Social Media",
    "facebook.com": "Social Media",
    "instagram.com": "Social Media",
    
    # College
    "srmist.edu.in": "College Tools",
    "srmit.edu.in": "College Tools",
    
    # Music
    "spotify.com": "Music",
    "music.youtube.com": "Music",
}

# Keyword patterns for fallback classification
KEYWORD_MAP = {
    "Programming & Dev": [
        "github", "stackoverflow", "gitlab", "code", "api", "npm", 
        "python", "javascript", "programming", "developer"
    ],
    "Video": [
        "youtube", "video", "playlist", "stream", "watch", "vimeo"
    ],
    "Shopping": [
        "amazon", "flipkart", "buy", "product", "cart", "checkout", 
        "shopping", "store"
    ],
    "Reading / Articles": [
        "medium", "blog", "article", "news", "story", "essay"
    ],
    "Docs": [
        "docs", "documentation", "manual", "guide", "reference"
    ],
    "Community": [
        "reddit", "discord", "forum", "discussion", "community"
    ],
    "Learning / Courses": [
        "course", "lecture", "udemy", "coursera", "khanacademy", 
        "tutorial", "learning"
    ],
    "News": [
        "news", "breaking", "report", "journalism", "current events"
    ],
    "Music": [
        "lyrics", "song", "music", "spotify", "album", "artist"
    ],
    "Math & CS Concepts": [
        "binary", "algorithm", "division", "computer science"
    ],
    "Social Media": [
        "twitter", "facebook", "instagram", "social", "post"
    ],
    "College Tools": [
        "srm", "college", "university", "exam", "student portal"
    ]
}

# -------------------------
# TF-IDF INITIALIZATION (Lazy)
# -------------------------

_tfidf_vectorizer = None
_category_matrix = None
_category_names = None


def _build_tfidf():
    """
    Build TF-IDF vectorizer and category matrix (called once on first use).
    """
    global _tfidf_vectorizer, _category_matrix, _category_names
    
    if _tfidf_vectorizer is not None:
        return  # Already initialized

    docs = []
    names = []

    for category, examples in CATEGORY_EXAMPLES.items():
        for example in examples:
            docs.append(example)
            names.append(category)

    _tfidf_vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        stop_words="english",
        ngram_range=NGRAM_RANGE
    )
    
    _category_matrix = _tfidf_vectorizer.fit_transform(docs)
    _category_names = names


# -------------------------
# HELPER FUNCTIONS
# -------------------------

def _preprocess_titles(titles: List[str]) -> str:
    """
    Normalize and join tab titles for TF-IDF processing.
    
    Preserves title frequency (unlike set deduplication) to maintain
    signal strength when multiple tabs have similar content.
    
    Args:
        titles: List of tab titles
        
    Returns:
        Normalized concatenated string
    """
    normalized = []
    for t in titles:
        if not t:
            continue
        # Convert to string and normalize
        text = str(t).lower().strip()
        if text:
            normalized.append(text)
    
    # Join with spaces, collapse multiple spaces
    combined = " ".join(normalized)
    return re.sub(r"\s+", " ", combined).strip()


def _domain_category(domain: str) -> Optional[str]:
    """
    Map domain to category using exact and suffix matching.
    
    Args:
        domain: Domain string (e.g., "github.com")
        
    Returns:
        Category name or None if no match
    """
    if not domain:
        return None
    
    domain = domain.lower()

    # Exact match (fastest)
    if domain in DOMAIN_CATEGORY_MAP:
        return DOMAIN_CATEGORY_MAP[domain]

    # Suffix match (handles subdomains like "api.github.com")
    for map_domain, category in DOMAIN_CATEGORY_MAP.items():
        if domain.endswith(map_domain):
            return category

    return None


def _keyword_fallback(text: str) -> str:
    """
    Classify text using keyword matching with word boundaries.
    
    Uses regex word boundaries to avoid false positives like:
    - "code" matching "decode" or "encoded"
    - "api" matching "capital" or "rapid"
    
    Args:
        text: Preprocessed title text
        
    Returns:
        Category name (defaults to "Miscellaneous")
    """
    text = text.lower()
    scores = {}

    for category, keywords in KEYWORD_MAP.items():
        count = 0
        for keyword in keywords:
            # Use word boundaries to match whole words only
            pattern = r'\b' + re.escape(keyword.strip()) + r'\b'
            if re.search(pattern, text):
                count += 1
        
        if count > 0:
            scores[category] = count

    # Return category with highest keyword matches
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]
    
    return "Miscellaneous"


# -------------------------
# MAIN LABELING FUNCTION
# -------------------------

def generate_cluster_name(
    titles: List[str], 
    domains: List[str] = None
) -> str:
    """
    Generate a semantic label for a cluster of tabs.
    
    Strategy cascade (in order of confidence):
    1. Domain majority (if 50%+ tabs share recognizable domain)
    2. TF-IDF similarity to category examples
    3. Keyword pattern matching
    4. Default to "Miscellaneous"
    
    Args:
        titles: List of tab titles
        domains: Optional list of domains (same length as titles)
        
    Returns:
        Category name string
    """
    # Initialize TF-IDF on first call
    _build_tfidf()

    # Validate inputs
    if not titles:
        return "Miscellaneous"

    # Preprocess titles
    clean_titles = _preprocess_titles(titles)
    
    if not clean_titles:
        return "Miscellaneous"

    # -------------------------
    # STRATEGY 1: Domain Majority
    # -------------------------
    if domains:
        # Map domains to categories
        mapped_categories = [
            cat for cat in (_domain_category(d) for d in domains) 
            if cat is not None
        ]

        if mapped_categories:
            # Count category frequencies
            category_counts = Counter(mapped_categories)
            most_common_cat, count = category_counts.most_common(1)[0]

            # Check if majority threshold met
            # Guard against division by zero
            if domains and (count / len(domains)) >= DOMAIN_MAJORITY_THRESHOLD:
                return most_common_cat

    # -------------------------
    # STRATEGY 2: TF-IDF Scoring
    # -------------------------
    try:
        # Transform titles into TF-IDF vector
        vec = _tfidf_vectorizer.transform([clean_titles])
        similarities = cosine_similarity(vec, _category_matrix)[0]

        # Aggregate scores by category (multiple examples per category)
        category_scores = defaultdict(list)
        for i, score in enumerate(similarities):
            category = _category_names[i]
            category_scores[category].append(float(score))

        # Average similarity scores per category for stability
        avg_scores = {
            cat: np.mean(scores) 
            for cat, scores in category_scores.items()
        }

        # Find best matching category
        best_category, best_score = max(avg_scores.items(), key=lambda x: x[1])

        # Return if confidence threshold met
        if best_score >= TFIDF_SCORE_THRESHOLD:
            return best_category

    except Exception as e:
        # TF-IDF can fail on edge cases - continue to fallback
        pass

    # -------------------------
    # STRATEGY 3: Keyword Fallback
    # -------------------------
    keyword_category = _keyword_fallback(clean_titles)
    if keyword_category:
        return keyword_category

    # -------------------------
    # STRATEGY 4: Default
    # -------------------------
    return "Miscellaneous"