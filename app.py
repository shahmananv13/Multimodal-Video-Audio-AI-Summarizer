import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai

import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv
load_dotenv()

import os

API_KEY=os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

st.set_page_config(
    page_title="Multimodal AI Agent- Video Summarizer",
    page_icon="🎥",
    layout="wide"
)
st.title("Phidata Video AI Summarizer Agent 🎥🎤🖬")
st.header("Powered by Gemini 2.0 Flash Exp")

@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video/Audio AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        markdown=True,
    )

multimodal_Agent=initialize_agent()

video_file = st.file_uploader(
    "Upload a video file", type=['mp4', 'mov', 'avi'], help="Upload a video for AI analysis"
)
audio_file = st.file_uploader(
    "Upload a audio file", type=['m4a'], help="Upload an audio for AI analysis"
)
if video_file or audio_file:
    video_path = ""
    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name
        st.video(video_path, format="video/mp4", start_time=0)

    audio_path = ""
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as temp_audio:
            temp_audio.write(audio_file.read())
            audio_path = temp_audio.name
        st.audio(audio_path, format="audio/m4a")

    user_query = st.text_area(
        "What insights are you seeking from the video/audio?",
        placeholder="Ask anything about the video/audio content. The AI agent will analyze and gather additional context if needed.",
        help="Provide specific questions or insights you want from the video."
    )

    if st.button("🔍 Analyze Media", key="analyze_media_button"):
        if not user_query:
            st.warning("Please enter a question or insight to analyze the media.")
        else:
            try:
                with st.spinner("Processing media and gathering insights..."):
                    # Check if an audio or video file is uploaded
                    if video_path:
                        media_path = video_path
                        media_type = "video"
                    elif audio_path:
                        media_path = audio_path
                        media_type = "audio"
                    else:
                        st.warning("Please upload an audio or video file.")
                        st.stop()

                    # Process the media file    
                    processed_media = genai.upload_file(media_path)
                    print(processed_media)
                    while processed_media.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_media = get_file(processed_media.name)
                    
                    # Define the analysis prompt
                    analysis_prompt = (
                        f"""
                        {user_query}
                        """
                    )
                    # Run the analysis using the multimodal agent
                    print(dir(multimodal_Agent.run()))
                    response = multimodal_Agent.run(analysis_prompt, audio=processed_media)

                st.subheader("Analysis Result")
                st.markdown(response.content)
            except Exception as error:
                st.error(f"An error occurred during analysis: {error}")
            finally:
                Path(media_path).unlink(missing_ok=True)
    else:
        st.info("Upload an audio or video file to begin analysis.")


st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)