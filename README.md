# 🎬 YT Chat — Ask Anything About Any YouTube Video

> An AI-powered chatbot that lets you have a conversation with any YouTube video using a full RAG pipeline.

**[🚀 Live Demo](https://uditya246.github.io/yt-chat)** · **[Backend API Docs](https://yt-chat-backend-jlp1.onrender.com/docs)**

> ⚠️ First load may take ~30 seconds as the free server wakes up from sleep.

---

## ✨ What it does

Paste any YouTube URL → the app fetches the transcript, indexes it into a vector database, and lets you ask questions in natural language. The AI **only answers from the video content** — no hallucinations, no made-up facts.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 4 Scout 17B via Groq |
| Embeddings | BAAI/bge-small-en-v1.5 (FastEmbed) |
| Vector Store | FAISS |
| RAG Framework | LangChain |
| Transcript Fetching | Supadata API |
| Backend | FastAPI (Python) |
| Frontend | Vanilla HTML / CSS / JS |
| Backend Hosting | Render (free tier) |
| Frontend Hosting | GitHub Pages (free) |

---

## 🧠 How it works

```
YouTube URL
    │
    ▼
Transcript Extraction (Supadata API)
    │
    ▼
Text Chunking (RecursiveCharacterTextSplitter)
chunk_size=1000, chunk_overlap=200
    │
    ▼
Embedding (BAAI/bge-small-en-v1.5 via FastEmbed)
    │
    ▼
FAISS Vector Index
    │
    ▼
User Question ──► Similarity Search (k=4 most relevant chunks)
    │
    ▼
Prompt Template + Retrieved Context
    │
    ▼
Llama 4 Scout via Groq ──► Answer
```

---

## 🚀 Run it locally

### 1. Clone the repo
```bash
git clone https://github.com/uditya246/yt-chat-backend.git
cd yt-chat-backend
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set environment variables
```bash
export HF_TOKEN=your_huggingface_token
export SUPADATA_API_KEY=your_supadata_key
```

### 4. Run the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5. Open the frontend
Open `index.html` in your browser and update `API_BASE` to `http://localhost:8000`

---

## 📡 API Endpoints

### `POST /load`
Fetch and index a YouTube video transcript.
```json
{ "url": "https://www.youtube.com/watch?v=VIDEO_ID" }
```
**Response:**
```json
{ "video_id": "VIDEO_ID", "message": "Video loaded successfully", "chunks": 24 }
```

### `POST /chat`
Ask a question about a loaded video.
```json
{ "video_id": "VIDEO_ID", "question": "What is this video about?" }
```
**Response:**
```json
{ "answer": "This video is about..." }
```

---

## 🔑 Environment Variables

| Variable | Description | Where to get it |
|---|---|---|
| `HF_TOKEN` | HuggingFace API token | huggingface.co/settings/tokens |
| `SUPADATA_API_KEY` | Transcript fetching API | supadata.ai |

---

## ⚙️ Deployment

- **Backend** → [Render.com](https://render.com) (free tier)
  - Build command: `pip install -r requirements.txt`
  - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Frontend** → GitHub Pages (free)
  - Just upload `index.html` and enable Pages in repo settings

---

## 👨‍💻 Author

Built by **Uditya**
- GitHub: [uditya246](https://github.com/uditya246)
- LinkedIn: [Uditya Yadav](https://www.linkedin.com/in/uditya-yadav-3480b1323/)

---

## 📄 License

MIT License — free to use and modify.
