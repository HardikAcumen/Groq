import streamlit as st
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage

from dotenv import load_dotenv
import os

load_dotenv() 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  


def AskQuestion_to_AI(): 
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """

    user_question = st.text_input("Type here..")
    # Get Groq API key
    llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY) # Replace 'your_api' with your actual API key


    response = llm.complete(user_question)

    st.write(response)

if __name__ == "__main__":
    AskQuestion_to_AI()