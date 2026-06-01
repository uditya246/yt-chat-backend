from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import warnings
warnings.filterwarnings('ignore')

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from youtube_transcript_api.proxies import WebshareProxyConfig, GenericProxyConfig
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

# ── Pre-load embedding model at startup ────────────────────────────────────
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
    import re
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
    """Try fetching transcript with multiple fallback strategies."""

    # Strategy 1: No proxy (sometimes works)
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id, languages=["en"])
        return " ".join(chunk.text for chunk in transcript_list)
    except Exception:
        pass

    # Strategy 2: Use Webshare proxy if credentials are set
    webshare_user = os.environ.get("WEBSHARE_USER", "")
    webshare_pass = os.environ.get("WEBSHARE_PASS", "")
    if webshare_user and webshare_pass:
        try:
            proxy_config = WebshareProxyConfig(
                proxy_username=webshare_user,
                proxy_password=webshare_pass,
            )
            api = YouTubeTranscriptApi(proxy_config=proxy_config)
            transcript_list = api.fetch(video_id, languages=["en"])
            return " ".join(chunk.text for chunk in transcript_list)
        except Exception:
            pass

    # Strategy 3: Generic HTTP proxy if set
    http_proxy = os.environ.get("HTTP_PROXY", "")
    if http_proxy:
        try:
            proxy_config = GenericProxyConfig(
                http_url=http_proxy,
                https_url=http_proxy,
            )
            api = YouTubeTranscriptApi(proxy_config=proxy_config)
            transcript_list = api.fetch(video_id, languages=["en"])
            return " ".join(chunk.text for chunk in transcript_list)
        except Exception:
            pass

    raise Exception(
        "YouTube is blocking requests from this server's IP. "
        "This is a common issue with cloud hosting. "
        "Please try a different video or set up a proxy (WEBSHARE_USER + WEBSHARE_PASS env vars)."
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

    # 1. Fetch transcript
    try:
        transcript = fetch_transcript(video_id)
    except TranscriptsDisabled:
        raise HTTPException(status_code=400, detail="No English captions available for this video.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # 3. Embed + index
    vector_store = FAISS.from_documents(chunks, EMBEDDINGS)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 4. Build chain
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
