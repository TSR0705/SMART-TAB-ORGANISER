# Smart Tab Organizer

<p align="center">
  <em>Automatically group your open browser tabs into meaningful categories using lightweight AI</em>
</p>

<p align="center">
  <a href="https://github.com/tsr0705/smart-tab-organizer/releases">
    <img src="https://img.shields.io/github/v/release/tsr0705/smart-tab-organizer?style=flat-square" alt="GitHub release">
  </a>
  <a href="https://github.com/tsr0705/smart-tab-organizer/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/tsr0705/smart-tab-organizer?style=flat-square" alt="License">
  </a>
  <a href="https://railway.app/">
    <img src="https://img.shields.io/badge/deployment-railway-blue?style=flat-square&logo=railway" alt="Deployment - Railway">
  </a>
  <a href="https://developer.chrome.com/docs/extensions/mv3/">
    <img src="https://img.shields.io/badge/chrome-extension-blue?style=flat-square&logo=googlechrome" alt="Chrome Extension">
  </a>
</p>

## 🧠 Overview

Smart Tab Organizer is a Chrome extension that automatically groups your open browser tabs into meaningful semantic categories using lightweight AI. No more manually organizing dozens of tabs - let AI do the work for you!

### Why Smart Tab Organizer?

Modern users keep 10-50+ tabs open simultaneously, leading to cognitive overload, lost tabs, and wasted time. Smart Tab Organizer solves this by intelligently grouping your tabs into categories like "Coding", "Research", "Shopping", and "Videos" so you can:

- Quickly find related tabs
- Close entire topic groups with one click
- Reduce tab bar clutter
- Improve browsing productivity

### Lightweight AI That Works Offline

Unlike other tab management tools, Smart Tab Organizer uses a lightweight AI pipeline that works completely offline:

- Local embeddings using Sentence Transformers (MiniLM)
- TF-IDF vectorization for semantic understanding
- Keyword-based heuristics for fallback classification
- No API keys or paid services required
- Works for any user, anywhere in the world

## 🌟 Features

- ✅ **AI-powered tab clustering** - Automatically groups semantically related tabs
- ✅ **Instant automatic grouping** - One-click organization of all open tabs
- ✅ **Local embeddings** - Uses MiniLM sentence transformers for fast, offline processing
- ✅ **Offline classifier** - TF-IDF + keyword heuristics ensure 100% offline operation
- ✅ **Clean UI** - Beautiful, intuitive interface with collapsible cluster cards
- ✅ **"Close All" actions** - Remove entire topic groups with a single click
- ✅ **Smart tab matching** - Robust URL matching that handles query parameters and redirects
- ✅ **Chrome extension + FastAPI backend** - Modern, maintainable architecture
- ✅ **Free deployment** - Deploy to Railway's free tier in minutes

## 🏗️ Architecture

```
┌──────────────────────────┐
│   Chrome Extension       │
│  (popup.js/background.js)│
└───────────┬──────────────┘
            │ REST API
            ▼
┌──────────────────────────┐
│     FastAPI Backend      │
│   (Python 3.11 + Docker) │
└───────────┬──────────────┘
            │ Lightweight AI
            ▼
┌────────────────────────────────────┐
│  Sentence Transformers Embeddings  │
│        + TF-IDF Labeler            │
└────────────────────────────────────┘
```

## 📁 Folder Structure

```
.
├── backend/
│   ├── app/
│   │   ├── cluster.py         # Embedding generation and clustering logic
│   │   ├── labeler.py         # TF-IDF-based labeling system
│   │   ├── schemas.py         # Pydantic models for request/response
│   │   └── main.py            # FastAPI application entry point
│   ├── Dockerfile             # Container configuration
│   └── requirements.txt       # Python dependencies
└── extension/
    ├── popup.html             # Extension UI
    ├── popup.js               # Frontend logic
    ├── background.js          # Background service worker
    ├── manifest.json          # Extension configuration
    ├── styles.css             # Styling
    └── icons/                 # Extension icons
```

## ⚙️ How It Works

1. **Tab Collection**: Chrome extension gathers all open tab titles and URLs
2. **Embedding Generation**: Backend generates semantic embeddings using MiniLM sentence transformers
3. **Clustering**: DBSCAN algorithm groups semantically similar tabs together
4. **Labeling**: TF-IDF vectorization assigns meaningful category names to clusters
5. **UI Rendering**: Extension displays organized clusters with "Close All" actions
6. **Privacy**: All processing happens locally - no data leaves your computer

The model choice ensures speed and zero cost:
- MiniLM is 5x faster than BERT with comparable performance
- TF-IDF requires no internet connection
- DBSCAN clustering adapts to the number of tabs automatically
- All models run locally with minimal RAM usage

## 🛠️ Installation (Local Development)

### Prerequisites
- Python 3.11+
- Google Chrome
- Node.js (for development tools)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/smart-tab-organizer.git
   cd smart-tab-organizer
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install backend dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Run the backend server**
   ```bash
   uvicorn app.main:app --reload
   ```

5. **Load the extension in Chrome**
   - Open Chrome and navigate to `chrome://extensions`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select the `extension` directory

## ☁️ Deployment Guide (Railway)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/your-username/smart-tab-organizer.git
   git push -u origin main
   ```

2. **Deploy to Railway**
   - Go to [Railway.app](https://railway.app/)
   - Create a new project
   - Connect your GitHub repository
   - Railway will automatically detect the Dockerfile and deploy

3. **Configure the extension**
   - Update the `BACKEND_URL` in `extension/popup.js` to your Railway deployment URL
   - Reload the extension in Chrome

## 📸 Usage

### Extension Popup
```
┌─────────────────────────────┐
│   Smart Tab Organizer       │
├─────────────────────────────┤
│  [ Cluster Tabs ]           │
├─────────────────────────────┤
│  🔧 Development (5)         │
│  ├─ GitHub - Project        │
│  ├─ Stack Overflow          │
│  ├─ Python Documentation    │
│  └─ [ Close All ]           │
│                             │
│  🛒 Shopping (3)            │
│  ├─ Amazon                  │
│  ├─ eBay                    │
│  └─ [ Close All ]           │
│                             │
│  🎥 Videos (4)              │
│  ├─ YouTube - Tutorial      │
│  ├─ Netflix                 │
│  └─ [ Close All ]           │
└─────────────────────────────┘
```

## 🧰 Tech Stack

| Component | Technology |
|----------|------------|
| Backend | [FastAPI](https://fastapi.tiangolo.com/) |
| AI Models | [Sentence Transformers](https://www.sbert.net/) |
| Clustering | [Scikit-learn DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan) |
| Text Processing | [TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) |
| Extension | [Chrome Extensions Manifest V3](https://developer.chrome.com/docs/extensions/mv3/) |
| Deployment | [Railway](https://railway.app/) |
| Containerization | [Docker](https://www.docker.com/) |

## ⚠️ Limitations & Future Improvements

### Current Limitations
- Clustering accuracy depends on tab title quality
- Limited to Chrome browser (for now)
- Basic category classification

### Future Improvements
- 🧠 **Advanced semantic grouping** - Fine-tune clustering with more sophisticated algorithms
- 🌐 **Multi-browser support** - Firefox, Safari, Edge extensions
- ⚡ **Offline WASM inference** - Compile models to WebAssembly for pure client-side processing
- 📚 **Bookmark integration** - Organize bookmarks using the same AI system
- 📈 **Persistent cluster learning** - Remember user preferences and improve over time

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code follows the existing style and includes appropriate tests.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👥 Credits

Built with ❤️ 
---
## 👤 About the Creator

<div align="center">
  <img src="https://avatars.githubusercontent.com/TSR0705" alt="Tanmay Singh" width="100" style="border-radius:50%;" />
  <h3>Tanmay Singh</h3>
<p>
  <em>Rising Full-Stack Innovator Shaping Next-Gen Web Experiences</em><br/>
  Cloud-First Mindset | UI/UX-Driven | JavaScript at the Core
</p>

Special thanks to:
- [Sentence Transformers](https://www.sbert.net/) for excellent embedding models
- [FastAPI](https://fastapi.tiangolo.com/) for the amazing backend framework
- [Scikit-learn](https://scikit-learn.org/) for clustering algorithms
