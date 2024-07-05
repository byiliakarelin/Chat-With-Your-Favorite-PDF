# Python code

import streamlit as st
import os
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from htmlUse import bot_template, user_template, css

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        huggingfacehub_api_token=api_token,
        model_kwargs={
            "temperature": 0.5,
            "max_new_tokens": 250, # or add "max_length": 512
        }
    )
    
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Helpful Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    return conversation_chain

st.set_page_config(page_title="Chat With Your Favorite PDF", page_icon="🏋️")

if "conversation" not in st.session_state:
    st.session_state.conversation = None

def display_message(role, message):
    if role == "user":
        with st.chat_message("user"):
            st.markdown(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)
    else:
        with st.chat_message("bot"):
            message_placeholder = st.empty()
            full_response = ""
            for word in message.split():
                full_response += word + " "
                time.sleep(0.4)  # Adjust this value to control the speed of word appearance
                message_placeholder.markdown(bot_template.replace("{{MSG}}", full_response), unsafe_allow_html=True)
            message_placeholder.markdown(bot_template.replace("{{MSG}}", full_response.strip()), unsafe_allow_html=True)

def display_message(role, message):
    if role == "user":
        st.markdown(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)
    else:
        st.markdown(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.header("Chat With Your Favorite PDF 🏋️")
    st.markdown(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Always display the input box at the top
    user_question = st.text_input("Ask a question")

    # Create a container for the chat history
    chat_container = st.container()

    # Process user input
    if user_question:
        if st.session_state.conversation:
            with st.spinner('Thinking very hard...'):
                try:
                    response = st.session_state.conversation.invoke({"question": user_question})
                    bot_response = response['answer']
                    # Add both question and answer as a tuple
                    st.session_state.chat_history.append((user_question, bot_response))
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        else:
            st.warning("Please process the documents first before asking questions.")

    # Display chat history in the container, with most recent messages at the top
    with chat_container:
        for user_q, bot_a in reversed(st.session_state.chat_history):
            display_message("user", user_q)
            display_message("bot", bot_a)

    with st.sidebar:
        st.subheader("Your important documents")
        pdf_docs = st.file_uploader("Upload your podcast:", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner('Processing documents...'):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.success("Documents processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
