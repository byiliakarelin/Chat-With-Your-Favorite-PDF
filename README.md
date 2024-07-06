# Chat-With-Your-Favorite-PDF

This is a public repository for a web application that allows you to chat with a PDF on Streamlit.
This application allows users to chat with their PDF documents using natural language processing and machine learning techniques while using LLM. 

## **Breakdown of how the code works:**
***
### **Necessary libraries:**
```
import streamlit as st # Streamlit is used for creating the web application interface
import os # Standard Python library for operating system functions
import time # Standard Python library for operating time-related functions
from dotenv import load_dotenv # Used to load environment variables from a .env file
from PyPDF2 import PdfReader # This is used for reading and extracting text from PDF files
from langchain.text_splitter import CharacterTextSplitter # This is used to split large texts into smaller chunks
from langchain_huggingface import HuggingFaceEmbeddings # This is used to create text embeddings using Hugging Face models
from langchain_community.vectorstores import FAISS # FAISS is used for efficient similarity search and clustering of dense vectors
from langchain.memory import ConversationBufferMemory # This is used to maintain conversation history
from langchain.chains import ConversationalRetrievalChain # This chain is used for question-answering tasks over a set of documents
from langchain.prompts import PromptTemplate # This is used to create templates for prompts to the language model
from langchain.llms import HuggingFaceHub # This is used to interact with language models hosted on Hugging Face's model hub
from htmlUse import bot_template, user_template, css # Custom HTML templates and CSS for styling the chat interface
```
The get_pdf_text function extracts text from uploaded PDF files:
```
def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
```
This function iterates through each page of each PDF and combines all the extracted text.

### **Text Chunking**

The get_text_chunks function splits the extracted text into smaller, manageable chunks:
```
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
```
This step is crucial for processing large amounts of text efficiently.

### **Vector Store Creation**

The get_vectorstore function creates a vector store from the text chunks:

```
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
```
This function uses the Hugging Face embeddings to convert text chunks into vector representations, which are then stored in a FAISS index for efficient similarity search.

### **Conversation Chain Setup**

The get_conversation_chain function sets up the conversation chain:

```
def get_conversation_chain(vectorstore):
    # ... (LLM setup)
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
```
This function creates a conversational retrieval chain using the Hugging Face model, the vector store, and a conversation memory.

### **Streamlit UI**

The code uses Streamlit to create the user interface:
```
st.set_page_config(page_title="Chat With Your Favorite PDF", page_icon="🏋️")
```


