# backend/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import whisper
import uuid
import os
import shutil
import tempfile

# --- LLM and RAG Dependencies ---
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# --- ElevenLabs TTS Dependencies ---
from elevenlabs.client import ElevenLabs

from dotenv import load_dotenv

# --- API Setup ---
app = FastAPI()

# --- Global In-Memory Stores ---
conversation_memory = {}
retriever = None  # initialized later
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)
# --- Model Loading ---
OUTPUTS_DIR = "outputs"
DATA_DIR = "data"
if not os.path.exists(OUTPUTS_DIR):
    os.makedirs(OUTPUTS_DIR)

whisper_model = whisper.load_model("base")

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set.")
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY is not set.")

elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.5, "max_length": 300}
)


# --- Upload PDF and build knowledge base ---
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global retriever

    # ✅ Ensure "data" folder exists
    os.makedirs(DATA_DIR, exist_ok=True)

    pdf_path = os.path.join(DATA_DIR, file.filename)
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    return JSONResponse(content={"message": f"PDF '{file.filename}' uploaded and knowledge base created."})


# --- Audio Upload & Transcription ---
@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        result = whisper_model.transcribe(tmp_path, word_timestamps=True)
        
        transcription_text = result["text"]

        timestamps = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
            for seg in result["segments"] if "start" in seg and "end" in seg
        ]

        conversation_memory[session_id] = {
            "transcription": transcription_text,
            "timestamps": timestamps,
            "history": []
        }

        return JSONResponse(
            content={
                "session_id": session_id,
                "transcription": transcription_text,
                "timestamps": timestamps
            }
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- Chat Endpoint ---
class Message(BaseModel):
    session_id: str = None
    message: str
    output_format: str = "text"  # "text" or "audio"


@app.post("/chat")
async def chat(message_body: Message):
    session_id = message_body.session_id or str(uuid.uuid4())
    message = message_body.message
    output_format = message_body.output_format

    # Create RAG chain
    if retriever:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=None
        )
        result = qa_chain.invoke({"question": message})
        answer = result["answer"]
    else:
        result = llm.invoke(message)
        answer = result.content if hasattr(result, 'content') else str(result)

    audio_url = None
    if output_format == "audio":
        session_dir = os.path.join(OUTPUTS_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        audio_path = os.path.join(session_dir, "response.wav")

        with open(audio_path, "wb") as f:
            audio_data = elevenlabs_client.generate(
                text=answer,
                voice="Rachel",
                model="eleven_monolingual_v1"
            )
            f.write(audio_data)

        audio_url = f"/outputs/{session_id}/response.wav"

    return JSONResponse(
        content={
            "session_id": session_id,
            "answer": answer,
            "citations": [],
            "audio_url": audio_url
        }
    )


@app.get("/outputs/{session_id}/{filename}")
async def get_audio_file(session_id: str, filename: str):
    file_path = os.path.join(OUTPUTS_DIR, session_id, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    return JSONResponse(content={"error": "File not found."}, status_code=404)


@app.get("/session/{session_id}")
async def get_session_transcript(session_id: str):
    if session_id in conversation_memory:
        return JSONResponse(content=conversation_memory[session_id])
    return JSONResponse(content={"error": "Session not found."}, status_code=404)
