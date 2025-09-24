# Multimodal Conversational Pipeline

This project builds a multimodal conversational pipeline capable of processing speech and text input, and generating text and speech output. It leverages a modern tech stack to create a seamless user experience.

## Features
-   **Speech-to-Text (ASR):** Transcribes audio files (`.wav`) using OpenAI's Whisper model.
-   **Text Processing (LLM):** Uses an open-source LLM (Mistral-7B) with LangChain for multi-turn conversation, grounded in a provided knowledge base (RAG) to prevent hallucinations.
-   **Text-to-Speech (TTS):** Synthesizes AI responses into audio using the Coqui TTS library.
-   **API:** A FastAPI backend provides clear, well-documented endpoints for audio upload and chat.
-   **UI:** A Streamlit frontend provides a user-friendly interface for easy interaction.
-   **Reproducibility:** The entire project is containerized with Docker, ensuring a consistent setup across any environment.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd multimodal_pipeline
    ```

2.  **Add your Knowledge Base:**
    Place your knowledge base file (e.g., `llm_notes.pdf`) and any sample audio files (`.wav`) in the `data/` directory.

3.  **Get a Hugging Face API Token:**
    -   Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
    -   Create a new token with "read" access.
    -   Add the token to the `.env` file in your project's root directory:
        ```
        HUGGINGFACEHUB_API_TOKEN="<your_token>"
        ```

4.  **Build and Run with Docker:**
    ```bash
    docker build -t multimodal-pipeline .
    docker run -p 8000:8000 multimodal-pipeline
    ```
    This will start the FastAPI backend on `http://localhost:8000`.

5.  **Run the Streamlit UI:**
    In a **separate terminal**, navigate to the project root and run:
    ```bash
    streamlit run app.py
    ```
    The Streamlit UI will open in your browser, connecting to the backend.

## API Endpoints

-   `POST /upload_audio`: Transcribes an audio file.
-   `POST /chat`: Handles multi-turn text conversations.
-   `GET /outputs/{session_id}/{filename}`: Serves the synthesized audio file.
-   `GET /session/{session_id}`: Retrieves the full session transcript and history.