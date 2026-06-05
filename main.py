from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
import json
import pickle
import warnings
import requests
warnings.filterwarnings('ignore')

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import InferenceClient

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="YT Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Storage directory (persists across restarts within same deploy) ─────────
STORE_DIR = "/tmp/yt_chat_store"
os.makedirs(STORE_DIR, exist_ok=True)

# ── Pre-load embedding model ───────────────────────────────────────────────
print("Loading embedding model...")
EMBEDDINGS = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
print("Embedding model loaded!")

# ── HuggingFace LLM client ─────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
client = InferenceClient(
    model="meta-llama/Llama-3.2-1B-Instruct",
    token=HF_TOKEN
)

# ── In-memory chain cache ──────────────────────────────────────────────────
chain_cache: dict = {}

# ── Prompt template ────────────────────────────────────────────────────────
prompt = PromptTemplate(
    template="""You are a helpful assistant. Always respond in English only, regardless of the language of the question.
Answer only from the provided transcript context.
If the context is insufficient, say you don't know.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

# ── Helpers ────────────────────────────────────────────────────────────────
def extract_video_id(url_or_id: str) -> str:
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$"
    ]
    for p in patterns:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    raise ValueError("Could not extract video ID from: " + url_or_id)


def save_transcript(video_id: str, transcript: str, chunks_count: int):
    """Save transcript to disk so it survives server restarts."""
    path = os.path.join(STORE_DIR, f"{video_id}.json")
    with open(path, "w") as f:
        json.dump({"transcript": transcript, "chunks": chunks_count}, f)


def load_transcript_from_disk(video_id: str):
    """Load transcript from disk if it exists."""
    path = os.path.join(STORE_DIR, f"{video_id}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def fetch_transcript(video_id: str) -> str:
    """Try multiple transcript fetching strategies."""

    # Strategy 1: Direct YouTube timedtext API
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        url = f"https://www.youtube.com/watch?v={video_id}"
        resp = requests.get(url, headers=headers, timeout=15)
        match = re.search(r'"captionTracks":\[.*?"baseUrl":"(.*?)"', resp.text)
        if match:
            caption_url = match.group(1).replace("\\u0026", "&")
            caption_resp = requests.get(caption_url, headers=headers, timeout=15)
            matches = re.findall(r'<text[^>]*>(.*?)</text>', caption_resp.text, re.DOTALL)
            if matches:
                import html
                transcript = " ".join(html.unescape(m.strip()) for m in matches)
                if len(transcript) > 100:
                    print("✓ Fetched via direct YouTube API!")
                    return transcript
    except Exception as e:
        print(f"Direct fetch failed: {e}")

    # Strategy 2: youtubetranscript.com
    try:
        url = f"https://youtubetranscript.com/?server_vid2={video_id}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            matches = re.findall(r'<text[^>]*>(.*?)</text>', response.text, re.DOTALL)
            if matches:
                import html
                transcript = " ".join(html.unescape(m.strip()) for m in matches)
                if len(transcript) > 100:
                    print("✓ Fetched via youtubetranscript.com!")
                    return transcript
    except Exception as e:
        print(f"youtubetranscript.com failed: {e}")

    # Strategy 3: Supadata API
    try:
        api_key = os.environ.get("SUPADATA_API_KEY", "")
        if api_key:
            url = "https://api.supadata.ai/v1/youtube/transcript"
            headers = {"x-api-key": api_key}
            params = {"videoId": video_id, "lang": "en", "text": "true"}
            response = requests.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                content = data.get("content", "")
                if isinstance(content, list):
                    content = " ".join(item.get("text", "") for item in content)
                if len(str(content)) > 100:
                    print("✓ Fetched via Supadata!")
                    return str(content)
    except Exception as e:
        print(f"Supadata failed: {e}")

    raise Exception(
        "Could not fetch transcript for this video. "
        "Please try a different video with English captions enabled."
    )


def build_chain(retriever):
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    def invoke_model(formatted_prompt):
        messages = [{"role": "user", "content": formatted_prompt.text}]
        response = client.chat_completion(messages, max_tokens=512, temperature=0.1)
        return response.choices[0].message.content

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })
    return parallel_chain | prompt | RunnableLambda(invoke_model) | StrOutputParser()


def get_or_build_chain(video_id: str, transcript: str):
    """Build chain from transcript, cache it in memory."""
    if video_id in chain_cache:
        return chain_cache[video_id]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    vector_store = FAISS.from_documents(chunks, EMBEDDINGS)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    chain = build_chain(retriever)
    chain_cache[video_id] = chain
    return chain

# ── Request models ─────────────────────────────────────────────────────────
class LoadRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    video_id: str
    question: str

# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "YT Chat API is running"}


@app.post("/load")
def load_video(req: LoadRequest):
    try:
        video_id = extract_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Check disk cache first
    cached = load_transcript_from_disk(video_id)
    if cached:
        # Rebuild chain from cached transcript
        get_or_build_chain(video_id, cached["transcript"])
        return {"video_id": video_id, "message": "Loaded from cache", "chunks": cached["chunks"]}

    # Fetch fresh transcript
    try:
        transcript = fetch_transcript(video_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    chunks_count = len(chunks)

    # Save to disk
    save_transcript(video_id, transcript, chunks_count)

    # Build and cache chain
    vector_store = FAISS.from_documents(chunks, EMBEDDINGS)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    chain = build_chain(retriever)
    chain_cache[video_id] = chain

    return {"video_id": video_id, "message": "Video loaded successfully", "chunks": chunks_count}


@app.post("/chat")
def chat(req: ChatRequest):
    # Try to restore from disk if not in memory
    if req.video_id not in chain_cache:
        cached = load_transcript_from_disk(req.video_id)
        if cached:
            get_or_build_chain(req.video_id, cached["transcript"])
        else:
            raise HTTPException(status_code=404, detail="Video not loaded. Please load the video first.")

    chain = chain_cache[req.video_id]
    try:
        answer = chain.invoke(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
