import os
import streamlit as st
from mental_health_assistant import generate_response
import time

st.set_page_config(
    page_title="AI-Powered Mental Health Assistant",
    page_icon=":brain:",
    layout="wide",
)

st.title("🤖 AI-Powered Mental Health Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_logo = "man.png"
assistant_logo = "assistant.png"

def display_latest_message():
    latest_message = st.session_state.chat_history[-1]
    if latest_message['role'] == "user":
        st.image(user_logo, width=30)
        st.markdown(latest_message['content'])
    else:
        st.image(assistant_logo, width=30)
        st.markdown(latest_message['content'])

def handle_user_input(user_input):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    display_latest_message()
    response = generate_response(user_input)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    display_latest_message()

user_input = st.chat_input("How are you feeling today?")

if user_input:
    handle_user_input(user_input)
