import streamlit as st
import google.generativeai as genai
import time
import io
from PIL import Image, ImageEnhance
import numpy as np
from rembg import remove
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

import os

# Load environment variables
# load_dotenv()

# Set up API keys from Streamlit secrets
GENAI_API_KEY = st.secrets["GENAI_API_KEY"]
STABILITY_API_KEY = st.secrets["STABILITY_API_KEY"]

# Configure Gemini API
genai.configure(api_key=GENAI_API_KEY)

# Initialize Stability API client
try:
    stability_api = client.StabilityInference(
        key=STABILITY_API_KEY,
        verbose=True,
    )
except Exception as e:
    st.error(f"Error initializing Stability API: {str(e)}")

def apply_image_effects(image, effects):
    img = image.copy()
    
    if effects.get('brightness', 0) != 0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.0 + effects['brightness'])
    
    if effects.get('contrast', 0) != 0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.0 + effects['contrast'])
    
    if effects.get('grayscale', False):
        img = img.convert('L')
    
    if effects.get('black_and_white', False):
        img = img.convert('L').point(lambda x: 0 if x < 128 else 255, '1')
    
    if effects.get('remove_background', False):
        img = remove(img)
    
    return img

def generate_image(prompt):
    if not prompt:
        st.error("Please provide a prompt for image generation")
        return None
        
    try:
        st.info(f"🎨 Generating image for prompt: {prompt}")
        
        # Set up the generation parameters
        answers = stability_api.generate(
            prompt=prompt,
            seed=int(time.time()),  # Random seed for variety
            steps=30,
            cfg_scale=7.0,
            width=512,
            height=512,
            samples=1,
            sampler=generation.SAMPLER_K_DPMPP_2M
        )

        # Process the generated image
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    st.warning("⚠️ The generated image was filtered due to content safety restrictions.")
                    return None
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))
                    # Resize image to a smaller size
                    img = img.resize((384, 384), Image.Resampling.LANCZOS)
                    return img
                    
        st.warning("No image was generated in the response")
        return None
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

# Streamlit UI
st.title("💬 AI Chatbot with Image Generation")

# Initialize session state for images
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.generated_images = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history and images
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message:
            # Display the main image
            st.image(message["image"], use_column_width=True, caption="Generated Image")
            
            # Create three columns for the controls
            st.markdown("### Image Controls")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Basic adjustments
                st.markdown("#### Basic Adjustments")
                effects = {}
                effects['brightness'] = st.slider("Brightness", -1.0, 1.0, 0.0, key=f"bright_{message['timestamp']}")
                effects['contrast'] = st.slider("Contrast", -1.0, 1.0, 0.0, key=f"contrast_{message['timestamp']}")
            
            with col2:
                # Color options
                st.markdown("#### Color Options")
                effects['grayscale'] = st.checkbox("Grayscale", key=f"gray_{message['timestamp']}")
                effects['black_and_white'] = st.checkbox("Black & White", key=f"bw_{message['timestamp']}")
            
            with col3:
                # Advanced options
                st.markdown("#### Advanced Options")
                effects['remove_background'] = st.checkbox("Remove Background", key=f"rembg_{message['timestamp']}")
                
                # Download options
                st.markdown("#### Download")
                
            # Apply effects
            edited_image = apply_image_effects(message["image"], effects)
            
            # Preview of edited image
            st.markdown("#### Preview")
            preview_col1, preview_col2 = st.columns(2)
            with preview_col1:
                st.image(edited_image, use_column_width=True, caption="Edited Image")
            with preview_col2:
                buf = io.BytesIO()
                edited_image.save(buf, format="PNG")
                st.download_button(
                    label="💾 Download Edited Image",
                    data=buf.getvalue(),
                    file_name="edited_image.png",
                    mime="image/png",
                    key=f"download_{message['timestamp']}"
                )
            
            # Add a separator between messages
            st.markdown("---")

# User input
user_input = st.chat_input("Type your message... (Use #image: for image generation)")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    if user_input.startswith("#image:"):
        # Image generation using Stability AI
        prompt = user_input[7:].strip()
        with st.spinner("🎨 Generating image using Stability AI..."):
            generated_image = generate_image(prompt)
            if generated_image:
                timestamp = time.time()
                # Display the generated image immediately
                with st.chat_message("assistant"):
                    st.markdown(f"✨ Here's your generated image for: **{prompt}**")
                    st.image(generated_image, caption="Generated Image", use_column_width=True)
                # Store in session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Here's your generated image for: {prompt}",
                    "image": generated_image,
                    "timestamp": timestamp
                })
            else:
                st.error("Failed to generate image. Please try again with a different prompt.")
    else:
        # Regular chat using Gemini Pro
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Generating...")

        bot_response = ""
        try:
            with st.spinner("Generating response with Gemini..."):
                model = genai.GenerativeModel("gemini-2.0-flash-exp-image-generation")
                response = model.generate_content(user_input)
                bot_response = response.text

        except Exception as e:
            st.error(f"Error: {str(e)}")
            bot_response = "Sorry, I couldn't process your request."

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            for i in range(1, 11):
                message_placeholder.markdown(bot_response[:int(len(bot_response) * (i / 10))] + "▌")
                progress_bar.progress(i / 10)
                time.sleep(0.1)
            message_placeholder.markdown(bot_response)

        progress_bar.empty()
        status_text.empty()
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
