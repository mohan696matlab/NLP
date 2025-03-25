import streamlit as st
import requests
from transformers import AutoTokenizer


@st.cache_resource
def load_tokenizer(model_name="meta-llama/Llama-3.2-3B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


tokenizer = load_tokenizer()

def update_chat_history(ai_response, user_query, chat_history):
    chat_history.extend([{"role": "user", "content": user_query},
                        {"role": "assistant", "content": ai_response},
                         ])
    return chat_history


# URL of your FastAPI server
API_URL = "http://127.0.0.1:8000/generate/"

# Initialize chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("LLaMA Chatbot")

# Input from user
prompt = st.text_input("Ask something:")

if st.button("Send"):
    if prompt:
        # Add the current user query to the chat history
        st.session_state.chat_history = update_chat_history(None, prompt, st.session_state.chat_history)
        
        # Send a POST request to the FastAPI server with the updated chat history
        message = tokenizer.apply_chat_template(st.session_state.chat_history, tokenize=False)
        response = requests.post(API_URL, json={"prompt": message})
        response_data = response.json()['response']  # Convert response to JSON

        # Update chat history with the assistant's response
        st.session_state.chat_history = update_chat_history(response_data, None, st.session_state.chat_history)

        # Display the full conversation
        st.write("Conversation History:")
        for entry in st.session_state.chat_history:
            st.write(f"{entry['role'].capitalize()}: {entry['content']}")

    else:
        st.write("Please enter a prompt.")
