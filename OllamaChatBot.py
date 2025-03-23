from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

template = (
    "You are a conversational assistant. Respond concisely and accurately to the user's questions. "
    "Ensure clarity and avoid unnecessary information.\n\n"
    "User: {user_message}\n"
    "Assistant:"
)

model = OllamaLLM(model="llama3.2")

def get_ollama_response(user_message):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    response = chain.invoke({"user_message": user_message})
    return response

def run_chatbot():
    st.title("Ollama 3.2 Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_enabled = st.checkbox("Enable Chatbot", value=True)

    if chat_enabled:
        user_message = st.text_input("Enter your message:")
        
        if st.button("Send"):
            if user_message.strip():
                st.session_state.messages.append({"role": "user", "content": user_message})
                response = get_ollama_response(user_message)
                st.session_state.messages.append({"role": "bot", "content": response})
                st.write(f"Ollama 3.2: {response}")  
            else:
                st.error("Please enter a message.")

        if st.checkbox("Show Chat History"):
            st.write("Chat History:")
            for message in st.session_state.messages:
                role = "You" if message["role"] == "user" else "Ollama 3.2"
                st.write(f"{role}: {message['content']}")
    else:
        st.write("Chatbot is disabled. Enable the chatbot to start chatting!")
