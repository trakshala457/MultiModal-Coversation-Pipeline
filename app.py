# app.py
# This is the Streamlit frontend. It provides a user-friendly interface
# for our multimodal conversational pipeline.

import streamlit as st
import requests
# We import `streamlit` to build the UI and `requests` to make
# HTTP calls to our FastAPI backend.
import uuid
import os

# Define the backend API endpoint
# This variable stores the URL of our FastAPI backend. The Streamlit app
# will use this to send requests.
BACKEND_URL = "http://localhost:8000"

# --- UI Configuration ---
st.set_page_config(page_title="Multimodal AI Assistant", layout="centered")
# This sets the page title and centers the content for a cleaner look.

st.title("🗣️ Multimodal AI Assistant")
st.markdown("Interact with the system using either speech or text.")
# These lines add a title and a brief description to the web page.

# --- Session Management ---
# We use `st.session_state` to store and persist data across user interactions.
# This is crucial for maintaining conversation history and linking messages to a session.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.info(f"New conversation started. Session ID: {st.session_state.session_id}")
# If a new user visits or the session is reset, we generate a new UUID and
# initialize an empty chat history list.

# Display conversation history
# This loop iterates through the chat history stored in the session state
# and displays each message, including the speaker and the message text.
with st.container():
    for speaker, text, audio_url in st.session_state.chat_history:
        if speaker == "User":
            st.markdown(f"**You:** {text}")
        elif speaker == "AI":
            st.markdown(f"**AI:** {text}")
            # If the AI response has an audio URL, we display an audio player.
            # `st.audio` handles playback of the .wav file.
            if audio_url:
                st.audio(f"{BACKEND_URL}{audio_url}", format="audio/wav")


# --- Audio Input Section ---
with st.container():
    st.header("Audio Input 🎙️")
    uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"], key="audio_uploader")
    # `st.file_uploader` provides a drag-and-drop widget for audio files.

    if uploaded_file:
        with st.spinner("Transcribing and processing audio..."):
            try:
                # --- Step 1: Transcribe Audio ---
                # We make a POST request to the FastAPI backend's `/upload_audio` endpoint.
                # `uploaded_file.getvalue()` gets the file content to send in the request.
                files = {"file": uploaded_file.getvalue()}
                response_transcribe = requests.post(f"{BACKEND_URL}/upload_audio", files=files)
                response_transcribe.raise_for_status() # Raise an exception for bad status codes

                transcription = response_transcribe.json()["transcription"]
                # We get the transcription from the JSON response and add it to the chat history.
                st.session_state.chat_history.append(("User", transcription, None))

                # --- Step 2: Send Transcribed Text to Chat ---
                # Next, we send the transcribed text to the `/chat` endpoint.
                # We include the `session_id` to maintain the conversation context.
                response_chat = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"session_id": st.session_state.session_id, "message": transcription}
                )
                response_chat.raise_for_status()

                result = response_chat.json()
                answer = result["answer"]
                audio_url = result.get("audio_url")

                st.session_state.chat_history.append(("AI", answer, audio_url))
                st.success("Audio processed successfully!")

            except requests.exceptions.RequestException as e:
                # This `except` block catches any network or API-related errors.
                st.error(f"Error communicating with the backend: {e}")
            except Exception as e:
                # This handles any other unexpected errors during processing.
                st.error(f"An unexpected error occurred: {e}")

# --- Text Input Section ---
with st.container():
    st.header("Text Chat 💬")
    user_input = st.text_input("Type your message here:", key="text_input")
    # `st.text_input` creates a simple text box for typing.

    if st.button("Send", key="send_button") and user_input:
        with st.spinner("Getting AI response..."):
            try:
                st.session_state.chat_history.append(("User", user_input, None))

                # We directly send the user's text to the `/chat` endpoint.
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"session_id": st.session_state.session_id, "message": user_input}
                )
                response.raise_for_status()

                result = response.json()
                answer = result["answer"]
                audio_url = result.get("audio_url")

                st.session_state.chat_history.append(("AI", answer, audio_url))

            except requests.exceptions.RequestException as e:
                st.error(f"Error communicating with the backend: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# --- Reset Button ---
if st.button("Start New Conversation"):
    # This button allows the user to clear the session and start fresh.
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    # `st.experimental_rerun()` forces the app to reload and reinitialize the state.
    st.experimental_rerun()