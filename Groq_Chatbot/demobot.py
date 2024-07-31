"""
This is sample chatbot deployed on streamlit.
"""

import streamlit as st
# import random

from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv
import os

load_dotenv() 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  


def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """

    # Get Groq API key
    llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY) # Replace 'your_api' with your actual API key

    user_question = st.text_input("Ask a question:")


    response = llm.complete(user_question)
    st.write(response)


if __name__ == "__main__":
    main()