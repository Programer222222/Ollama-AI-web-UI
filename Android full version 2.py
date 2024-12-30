import requests
import json
import gradio as gr
from gtts import gTTS
import os
from playsound import playsound
import threading
from datetime import datetime
import sounddevice as sd
import wavio
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr  # Added for speech-to-text transcription

# Define the API endpoints and models
OLLAMA_BASE_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral"
WIZARDLM_MODEL = "wizardlm-uncensored:latest"
NEW_MODELS = ["model_1", "model_2", "wizardlm-uncensored:latest"]  # Add any new models here

# Set up headers for the request
headers = {
    'Content-Type': 'application/json',
}

# Initialize conversation history
conversation_history = []

# Define character context, images, and paths
CHARACTER_CONTEXTS = {
    "assistant": {
        "context": "You are a helpful assistant, here to provide information and support.",
        "image": r"C:\Users\Administrator\AppData\Local\Programs\Ollama\Android\AI chatbot 1.jpeg", 
    },
    "wizard": {
        "context": "You are a wise wizard, sharing knowledge and casting spells.",
        "image": r"C:\Users\Administrator\AppData\Local\Programs\Ollama\Android\Charachter 2.jpeg",
    },
    "robot": {
        "context": "You are a friendly robot, always ready to assist with tasks.",
        "image": r"C:\Users\Administrator\AppData\Local\Programs\Ollama\Android\Character 3.jpeg",
    },
    "cool_ai": {
        "context": "You are a cool AI, engaging and fun while providing assistance.",
        "image": r"C:\Users\Administrator\AppData\Local\Programs\Ollama\Android\Cool AI.jpeg",
    }
}

# Function to convert text to speech and play it
def speak_text(text, speed=1.0):
    tts = gTTS(text=text, lang='en', slow=speed < 1.0)
    temp_filename = "temp_audio.mp3"
    tts.save(temp_filename)

    def play_audio():
        playsound(temp_filename)
        os.remove(temp_filename)

    # Ensure audio is played without blocking other operations
    threading.Thread(target=play_audio).start()

# Function to generate a response using the selected model
def generate_response(prompt, model=DEFAULT_MODEL, temperature=0.7, character="assistant"):
    conversation_history.append(prompt)
    full_prompt = "\n".join(conversation_history)

    # Incorporate character context into the prompt
    character_context = CHARACTER_CONTEXTS.get(character, {}).get("context", "")
    full_prompt = f"{character_context}\n{full_prompt}"

    data = {
        "model": model,
        "stream": False,
        "prompt": full_prompt,
        "temperature": temperature,
    }

    try:
        response = requests.post(OLLAMA_BASE_URL, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            response_text = response.json()
            actual_response = response_text.get("response", "No response received.")
            conversation_history.append(actual_response)
            speak_text(actual_response)
            return actual_response
        else:
            error_message = f"Error: {response.status_code} - {response.text}"
            return error_message
    except requests.exceptions.RequestException as e:
        return f"API request failed: {e}"

# Reset conversation history
def reset_conversation():
    global conversation_history
    conversation_history = []
    return "Conversation reset."

# Function to record audio
def record_audio(duration):
    fs = 44100
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    audio_filename = f"user_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    wavio.write(audio_filename, recording, fs, sampwidth=2)
    return audio_filename, recording

# Function to play audio
def play_audio_file(file_path):
    if os.path.exists(file_path):
        playsound(file_path)

# Function to plot waveform
def plot_waveform(audio_data, fs=44100):
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(audio_data) / fs, len(audio_data)), audio_data)
    plt.title("Audio Waveform")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plot_filename = "audio_waveform.png"
    plt.savefig(plot_filename)
    plt.close()
    return plot_filename

# Function to transcribe audio
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        transcription = recognizer.recognize_google(audio_data)
        return transcription
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"

# Define the Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("## Enhanced Interactive AI Chatbot")

    # Character selection dropdown
    character_selector = gr.Dropdown(
        choices=list(CHARACTER_CONTEXTS.keys()),
        value="assistant",  # Default character
        label="Select Character"
    )

    # Display character context
    character_context_display = gr.Textbox(label="Character Context", interactive=False)

    with gr.Row():
        prompt_input = gr.Textbox(lines=2, placeholder="Enter your prompt here...", label="Text Prompt")

    with gr.Row():
        record_duration = gr.Slider(minimum=1, maximum=10, step=1, label="Recording Duration (seconds)")
        record_button = gr.Button("Record Audio")
        audio_playback_button = gr.Button("Play Recorded Audio")
        audio_submit_button = gr.Button("Submit Recorded Audio")
        recorded_audio_path = gr.Textbox(visible=False)
        audio_waveform_output = gr.Image(label="Audio Waveform")
        transcription_output = gr.Textbox(label="Transcription Output", interactive=False)

    # Model selection dropdown with new models dynamically added
    model_selector = gr.Dropdown(
        choices=[DEFAULT_MODEL, WIZARDLM_MODEL] + NEW_MODELS,
        value=DEFAULT_MODEL,
        label="Select Model"
    )

    # Temperature slider
    temperature_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, label="Temperature (Creativity Level)")

    response_output = gr.Textbox(label="AI Response", interactive=False)

    # Character image output
    character_image_output = gr.Image(label="Character Image", interactive=False)

    submit_button = gr.Button("Submit Text Prompt")
    reset_button = gr.Button("Reset Conversation")

    # Add PayPal donation button
    gr.HTML("""<h3>If you found this chatbot useful, consider donating via PayPal:</h3>
    <a href="https://www.paypal.me/ARansome63" target="_blank">
        <img src="https://www.paypalobjects.com/webstatic/icon/pp258.png" alt="Donate with PayPal" />
    </a>
    """)

    def handle_record(duration):
        file_path, audio_data = record_audio(duration)
        waveform_image = plot_waveform(audio_data)
        return file_path, waveform_image

    def send_transcription_to_prompt(transcription):
        # Update the prompt with the transcription
        return transcription

    def update_character_image(character):
        character_image_path = CHARACTER_CONTEXTS.get(character, {}).get("image", "")
        return character_image_path, CHARACTER_CONTEXTS.get(character, {}).get("context", "")

    submit_button.click(generate_response, inputs=[prompt_input, model_selector, temperature_slider, character_selector], outputs=response_output)
    reset_button.click(reset_conversation, outputs=response_output)

    record_button.click(handle_record, inputs=record_duration, outputs=[recorded_audio_path, audio_waveform_output])
    audio_playback_button.click(play_audio_file, inputs=recorded_audio_path, outputs=[])
    audio_submit_button.click(transcribe_audio, inputs=recorded_audio_path, outputs=transcription_output)

    transcription_to_prompt_button = gr.Button("Send Transcription to Prompt")
    transcription_to_prompt_button.click(send_transcription_to_prompt, inputs=transcription_output, outputs=prompt_input)

    # Update character image and context whenever a new character is selected
    character_selector.change(update_character_image, inputs=character_selector, outputs=[character_image_output, character_context_display])

    iface.launch(share=True, debug=True)
