import streamlit as st
import google.generativeai as genai
import time
# from dotenv import load_dotenv
import os

# Load environment variables
# load_dotenv()

# Set up Gemini API
GENAI_API_KEY = st.secrets["GENAI_API_KEY"]
genai.configure(api_key=GENAI_API_KEY)

# Streamlit UI
st.title("💬 Chatbot")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Show progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Generating...")

    # Try to use streaming first
    bot_response = ""
    try:
        with st.spinner("Generating response..."):
            model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")  # Use a valid model version
            response = model.generate_content(user_input)

        # Handle non-streaming response
        bot_response = response.text

    except Exception as e:
        st.error(f"Error: {str(e)}")
        bot_response = "Sorry, I couldn't process your request."

    # Display the response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Simulate streaming effect with progress bar
        for i in range(1, 11):
            message_placeholder.markdown(bot_response[:int(len(bot_response) * (i / 10))] + "▌")
            progress_bar.progress(i / 10)
            time.sleep(0.1)

        message_placeholder.markdown(bot_response)

    # Remove progress bar after completion
    progress_bar.empty()
    status_text.empty()

    # Store the bot's response in chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
