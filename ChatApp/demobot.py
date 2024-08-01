"""
This is sample chatbot deployed on streamlit.
"""

import streamlit as st
# import random
from llama_index.llms.groq import Groq
# from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv
import os
from AI.API import chat_API

load_dotenv() 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  


def main():
    user_question = st.text_input("Ask a question:")


    response = chat_API(user_question)
    st.write(response)


if __name__ == "__main__":
    main()