"""
This is sample chatbot deployed on streamlit.
"""

import streamlit as st
from AI.API import chat_API



def main():
    user_question = st.text_input("Ask a question:")


    response = chat_API(user_question)
    st.write(response)


if __name__ == "__main__":
    main()