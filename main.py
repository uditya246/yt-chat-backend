from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import warnings
import re
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

# ── In-memory store ────────────────────────────────────────────────────────
video_store: dict = {}

# ── Prompt template ────────────────────────────────────────────────────────
prompt = PromptTemplate(
    template="""You are a helpful assistant.
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


def fetch_transcript_supadata(video_id: str) -> str:
    """Fetch transcript using Supadata API — works from cloud IPs."""
    api_key = os.environ.get("SUPADATA_API_KEY", "")
    if not api_key:
        raise Exception("SUPADATA_API_KEY not set")
    
    url = f"https://api.supadata.ai/v1/youtube/transcript"
    headers = {"x-api-key": api_key}
    params = {"videoId": video_id, "lang": "en", "text": "true"}
    
    response = requests.get(url, headers=headers, params=params, timeout=30)
    if response.status_code == 200:
        data = response.json()
        # Supadata returns content as plain text or list
        content = data.get("content", "")
        if isinstance(content, list):
            return " ".join(item.get("text", "") for item in content)
        return str(content)
    raise Exception(f"Supadata API error: {response.status_code} {response.text}")


def fetch_transcript_youtubetranscript(video_id: str) -> str:
    """Fetch transcript using youtubetranscript.com API."""
    url = f"https://youtubetranscript.com/?server_vid2={video_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15)
    if response.status_code == 200:
        # Parse the XML/text response
        text = response.text
        # Extract text between <text> tags
        matches = re.findall(r'<text[^>]*>(.*?)</text>', text, re.DOTALL)
        if matches:
            import html
            return " ".join(html.unescape(m.strip()) for m in matches)
    raise Exception("youtubetranscript.com fetch failed")


def fetch_transcript_direct(video_id: str) -> str:
    """Fetch transcript directly from YouTube timedtext API."""
    # Get video page to find caption track URL
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    url = f"https://www.youtube.com/watch?v={video_id}"
    resp = requests.get(url, headers=headers, timeout=15)
    
    # Extract caption URL from page source
    match = re.search(r'"captionTracks":\[.*?"baseUrl":"(.*?)"', resp.text)
    if not match:
        raise Exception("No caption tracks found in video page")
    
    caption_url = match.group(1).replace("\\u0026", "&")
    
    # Fetch the caption XML
    caption_resp = requests.get(caption_url, headers=headers, timeout=15)
    caption_text = caption_resp.text
    
    # Extract text
    matches = re.findall(r'<text[^>]*>(.*?)</text>', caption_text, re.DOTALL)
    if not matches:
        raise Exception("No text found in captions")
    
    import html
    return " ".join(html.unescape(m.strip()) for m in matches)


def fetch_transcript(video_id: str) -> str:
    """Try multiple transcript fetching strategies."""

    # Strategy 1: Direct YouTube timedtext API (no external service needed)
    try:
        transcript = fetch_transcript_direct(video_id)
        if len(transcript) > 100:
            print("✓ Fetched via direct YouTube API!")
            return transcript
    except Exception as e:
        print(f"Direct fetch failed: {e}")

    # Strategy 2: Supadata API (free tier available)
    try:
        transcript = fetch_transcript_supadata(video_id)
        if len(transcript) > 100:
            print("✓ Fetched via Supadata!")
            return transcript
    except Exception as e:
        print(f"Supadata failed: {e}")

    # Strategy 3: youtubetranscript.com
    try:
        transcript = fetch_transcript_youtubetranscript(video_id)
        if len(transcript) > 100:
            print("✓ Fetched via youtubetranscript.com!")
            return transcript
    except Exception as e:
        print(f"youtubetranscript.com failed: {e}")

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

    if video_id in video_store:
        return {"video_id": video_id, "message": "Already loaded", "chunks": video_store[video_id]["chunks"]}

    try:
        transcript = fetch_transcript(video_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    vector_store = FAISS.from_documents(chunks, EMBEDDINGS)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    chain = build_chain(retriever)
    video_store[video_id] = {"chain": chain, "chunks": len(chunks)}

    return {"video_id": video_id, "message": "Video loaded successfully", "chunks": len(chunks)}


@app.post("/chat")
def chat(req: ChatRequest):
    if req.video_id not in video_store:
        raise HTTPException(status_code=404, detail="Video not loaded. Call /load first.")

    chain = video_store[req.video_id]["chain"]
    try:
        answer = chain.invoke(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
