from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
import html
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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── HuggingFace LLM client ─────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
client = InferenceClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    token=HF_TOKEN
)

# ── Lazy embedding model ───────────────────────────────────────────────────
_EMBEDDINGS = None
def get_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        print("Loading embedding model...")
        _EMBEDDINGS = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        print("Embedding model loaded!")
    return _EMBEDDINGS

# ── In-memory store ────────────────────────────────────────────────────────
video_store: dict = {}

# ── Prompt ─────────────────────────────────────────────────────────────────
prompt = PromptTemplate(
    template="""You are a helpful assistant. Always respond in English only.
Answer only from the provided transcript context.
If the context is insufficient, say "I don't know based on the video transcript."

Context:
{context}

Question: {question}

Answer in English:""",
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


def fetch_transcript(video_id: str) -> str:

    # Strategy 1: Supadata API (most reliable for cloud servers)
    supadata_key = os.environ.get("SUPADATA_API_KEY", "")
    if supadata_key:
        try:
            url = "https://api.supadata.ai/v1/youtube/transcript"
            headers = {"x-api-key": supadata_key}
            params = {"videoId": video_id, "lang": "en", "text": "true"}
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            print(f"Supadata response: {resp.status_code} {resp.text[:200]}")
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("content", "")
                if isinstance(content, list):
                    transcript = " ".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                    )
                else:
                    transcript = str(content)
                if len(transcript) > 100:
                    print("✓ Fetched via Supadata!")
                    return transcript
        except Exception as e:
            print(f"Supadata failed: {e}")

    # Strategy 2: Direct YouTube timedtext
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
                transcript = " ".join(html.unescape(m.strip()) for m in matches)
                if len(transcript) > 100:
                    print("✓ Fetched via direct YouTube!")
                    return transcript
    except Exception as e:
        print(f"Direct failed: {e}")

    raise Exception(
        "Could not fetch transcript. Please try a different video with English captions."
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

    vector_store = FAISS.from_documents(chunks, get_embeddings())
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
