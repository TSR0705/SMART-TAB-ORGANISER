---
applyTo: '**'
---
Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.

PROJECT NAME:
AI Tab Clusterer – Chrome Extension + Dockerized FastAPI ML Backend

GOAL:
Build a Chrome extension that automatically organizes all open browser tabs into semantic topic groups using AI embeddings and KMeans clustering.

SYSTEM OVERVIEW:
1. Chrome extension collects open tabs (title, URL, tabId).
2. Sends tab data to FastAPI backend (`POST /cluster`).
3. Backend generates embeddings using OpenAI model `text-embedding-3-small`.
4. Runs KMeans clustering on embeddings.
5. Applies heuristic logic to generate meaningful cluster names (Coding, Video, Shopping, Research, Blogs, Docs, etc.).
6. Returns structured JSON with clusters and their tabs.
7. Extension UI displays clusters in popup, lets user close all tabs inside any cluster.

TECH STACK:
Frontend (Chrome Extension):
- Manifest V3
- JavaScript (vanilla)
- popup.html, popup.js, background.js
- Tabs API, Storage API, Messaging API

Backend:
- Python 3.11
- FastAPI
- scikit-learn (KMeans)
- OpenAI Python SDK (embeddings)
- Pydantic
- Uvicorn
- Docker (containerized backend)

ARCHITECTURE:
root/
  backend/
    app/
      main.py
      cluster.py
      schemas.py
    requirements.txt
    Dockerfile
  extension/
    manifest.json
    popup.html
    popup.js
    background.js
    styles.css

REQUIREMENTS:
- Clean, modular, production-friendly code.
- Complete files (not partial snippets).
- Clear separation: Extension makes no clustering decisions; backend does all AI.
- Use async FastAPI endpoints.
- Use Pydantic models for request/response.
- KMeans cluster count chosen automatically (simple heuristic is fine).
- CORS must allow Chrome extension origins (localhost & chrome-extension://).
- No hardcoding of API keys in code. Read OPENAI_API_KEY from environment.

EXPECTED BEHAVIOR:
- Extension fetches all tabs → sends to backend → displays clusters.
- Clusters should group tabs semantically, not by domain only.
- “Close All Tabs” button closes all tabs belonging to a cluster.
- Errors shown in popup (network failure, backend down, missing API key).



CODING GUIDELINES FOR AI ASSISTANT:

1. ALWAYS generate complete file contents when asked.
   Never output partial code or "...".

2. Match folder structure exactly as defined.
   No extra directories unless specifically requested.

3. Code must be clean, readable, modular, with comments explaining logic.

4. For Python:
   - Use FastAPI + Pydantic.
   - Use type hints everywhere.
   - Use async endpoints.
   - Import only what is needed.
   - Raise proper HTTP exceptions on invalid input.
   - Add docstrings for all functions and classes.

5. For AI embeddings:
   - Use OpenAI embeddings with model "text-embedding-3-small".
   - Create a helper function to compute embeddings.
   - Do not store API keys in code.

6. For clustering:
   - Use scikit-learn KMeans.
   - Use dynamic cluster count heuristic:
       k = min(6, max(2, int(sqrt(n_tabs))))
   - Convert numpy arrays to Python lists before returning JSON.

7. For Chrome Extension:
   - manifest.json must be Manifest V3 compliant.
   - popup.js must handle:
       - loading spinner
       - fetch(tab list) → POST → JSON response
       - rendering clusters
       - close tabs via chrome.tabs.remove()
   - background.js handles privileged tab operations.
   - Use clean DOM manipulation (no frameworks).

8. For Docker:
   - Use python:3.11-slim
   - Keep image minimal
   - Expose port 8000
   - Use uvicorn to serve app

9. Error Handling:
   - Backend should handle missing titles/urls gracefully.
   - Frontend should show errors in UI.
   - Never crash silently.

10. When reviewing code:
   - Point out structural issues.
   - Ensure all endpoints, files, and integration points match the architecture.
   - Suggest safer, clearer, more maintainable patterns where applicable.

11. When suggesting changes:
   - Always provide the corrected full file or unified patch.
   - Keep compatibility with the rest of the system.

12. Final rule:
   BUILD A REAL, WORKING PRODUCT. No pseudo-code. No placeholders.




idea of the project 

🚀 AI TAB CLUSTERER — Complete, Detailed Project Concept
🎯 ONE-LINE SUMMARY

A Chrome extension that automatically groups your open browser tabs into meaningful categories using AI embeddings + clustering, and gives you one-click tools to clean or manage them.

🧠 WHAT PROBLEM DOES THIS SOLVE?

Modern users keep 10–50 tabs open (sometimes 100+).
This leads to:

cognitive overload

lost tabs

wasted time switching between topics

overwhelming tab bars

difficulty remembering which tab belongs to which task

huge RAM usage

ZERO browsers provide smart automatic tab grouping.

Your project fills that gap.

💡 CORE IDEA

The extension extracts all open tab titles + URLs → sends them to an AI backend → backend generates embeddings → performs clustering → returns clusters → extension shows clean groups like:

Coding

Research

Shopping

Videos

Documentation

Social Media

Tools

Blogs

You can then:

expand/collapse groups

see all tabs inside

close a whole group with one click

jump to a specific tab

reorder or rename groups

This feels magical, looks smart, and is actually useful.

🔥 WHY THIS IDEA IS UNIQUE

Most AI browser extensions do:

❌ summarization
❌ translation
❌ grammar fixes
❌ screenshots
❌ notes
❌ page analysis

NO ONE touches tab management using AI + clustering.
This is genuinely fresh and advanced.

You’re combining:

browser APIs

embeddings

ML clustering

smart heuristic naming

UI/UX

automation

This is not “baby AI”.
This looks like something a senior engineer built.

🧩 HOW IT WORKS — FULL PIPELINE (End-to-End)

Below is the exact sequence of events every time the user clicks “Cluster Tabs”.

1) Chrome Extension (frontend)
It does 4 things:
1. Get all open tabs

Using:

chrome.tabs.query({})


It retrieves:

tab title

tab URL

tab ID

2. Send to backend

It POSTs data:

{
  "tabs": [
    {"title": "GitHub - Project", "url": "https://github.com/..."},
    {"title": "React docs", "url": "https://react.dev/..."},
    {"title": "Amazon", "url": "https://amazon.in/..."}
  ]
}

3. Receive cluster data

Backend responds:

{
  "clusters": [
    {
       "id": 0,
       "name": "Coding",
       "tabs": [...]
    },
    {
       "id": 1,
       "name": "Shopping",
       "tabs": [...]
    }
  ]
}

4. Render UI + buttons

The popup displays:

Coding (5)
 - GitHub
 - StackOverflow
 - NPM
 [Close All]

Shopping (3)
 - Amazon
 - Flipkart
 [Close All]


Clicking “Close All” closes all tabs in that cluster.

2) Backend API (FastAPI + Docker)
It does REAL AI work:
STEP A — Turn tab text into embeddings

Using OpenAI:

each tab title + URL is converted into a numerical embedding vector (1536-dim)

embeddings capture semantics (“GitHub”, “StackOverflow”, “Docker docs” = related)

STEP B — KMeans Clustering

We run:

KMeans → groups mathematically similar embeddings

dynamic cluster count (usually 3–8 clusters)

STEP C — Name clusters

We infer category based on URLs:

Contains “github”, “stack”, “npm”, “docs” → Coding

Contains “youtube”, “watch” → Video

Contains “amazon”, “flipkart” → Shopping

Contains “medium”, “blog” → Reading

Contains “linkedin”, “twitter” → Social

Contains “google docs”, “drive” → Docs

If no match:

fallback: “Topic 1”, “Topic 2”

STEP D — Return structured JSON

FastAPI returns clean cluster results.

🏛️ SYSTEM ARCHITECTURE (Exact Components)
┌──────────────────────────┐
│      Chrome Extension    │
│  popup.html / popup.js   │
└───────────┬──────────────┘
            │ fetch tabs
            ▼
┌──────────────────────────┐
│    FastAPI Backend       │
│  /cluster endpoint       │
└───────────┬──────────────┘
            │ send tab metadata
            ▼
┌──────────────────────────┐
│       AI Pipeline        │
│  Embeddings + KMeans     │
└───────────┬──────────────┘
            │ return clusters
            ▼
┌──────────────────────────┐
│      Chrome Extension    │
│     Render UI + actions  │
└──────────────────────────┘

🎨 UI/UX Behavior
Popup UI elements:

Header: AI Tab Clusterer

Button: “Cluster Tabs”

Loading spinner

Cluster cards:

cluster name

tab list (clickable)

“Close All” button

Footer: “Powered by AI”

Design style:

clean

light shadows

simple cards

scrollable

clear spacing

🛠️ TECH STACK (Final, Defined)
Frontend

JavaScript (vanilla)

HTML

CSS

Chrome Extension Manifest V3

Chrome Tabs API

Backend

Python 3.11

FastAPI

Uvicorn

OpenAI Embeddings

Scikit-learn KMeans

Pydantic models

Docker

Optional Future

Redis / PostgreSQL (cache)

HuggingFace embeddings (offline)

Local WASM embedding models

📦 FEATURES (Detailed)
1. Auto cluster tabs with AI

Detects topics like:

Coding

Shopping

Research

Blogs

Music

Social Media

2. Close entire groups

One click → clean your browser.

3. Rename groups (optional extension)

You can rename “Topic 2” → “Machine Learning”.

4. Save cluster presets (future)

Automatically reclose known distractions.

5. Auto-tag new tabs (future)

When a new tab is opened → classify it instantly.

⚠️ LIMITATIONS

Be realistic:

If titles are unclear (“Home”, “Login”) embeddings may fail.

Backend requires internet (OpenAI).

Latency depends on embedding API speed.

CORS must be configured correctly.

Tab data is NOT stored — but users may still worry about privacy.

KMeans may occasionally mis-group tabs (normal for unsupervised ML).

🌱 FUTURE UPGRADES (You can add later)

Local embeddings (no API) using sentence-transformers WASM

ML category classifier trained on thousands of URLs

Browser-side clustering (no backend)

Sync with Chrome’s built-in tab groups

Dark mode UI

Analytics page inside extension

Automatically detect “focus mode” clusters

🧨 Potential Viral Impact

This idea is VERY publishable:

You can make a YouTube video demo

Share extension on GitHub

Get stars → good portfolio

Post on Reddit r/ChromeExtensions, r/webdev

Write blog: “I built an AI that organizes your tabs automatically”

Recruiters will love it

Even non-tech users love productivity tools

It looks like you built a startup, not a toy project.