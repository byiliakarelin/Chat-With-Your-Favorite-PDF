# Chat-With-Your-Favorite-PDF

This is a public repository for a web application that allows you to chat with a PDF on Streamlit, with ~150 lines of Python code.

This PDF Chat Application is a tool that allows users to interact with their PDF documents using natural language queries. By leveraging Large Language Models (LLMs), the application processes uploaded PDFs and enables users to ask questions about the content, and get responses based on the document's information. This project combines the power of ML with a chat interface, making it easier to extract insights from PDF documents.

In my case, I love Super Data Science podcast that is hosted by John Krohn. On the picture below, you can see that I asked who the guest was on episode 791.
<img width="1179" alt="Screen Shot 2024-07-07 at 5 23 15 PM" src="https://github.com/byiliakarelin/Chat-With-Your-Favorite-PDF/assets/132295797/2180a1e6-be0f-4c85-8741-967201c2933e">
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
***
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
***
### **Vector Store Creation**

The get_vectorstore function creates a vector store from the text chunks:

```
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
```
This function uses the Hugging Face embeddings to convert text chunks into vector representations, which are then stored in a FAISS index for efficient similarity search.
***
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
***
### **Streamlit UI**

The code uses Streamlit to create the user interface:
```
st.set_page_config(page_title="Chat With Your Favorite PDF", page_icon="🏋️")
```
***
### **Main Function**

The main function ties everything together:

- It sets up the Streamlit interface.
- Handles PDF uploads in the sidebar.
- Processes the PDFs when the "Process" button is clicked.
- Displays the chat history.
- Handles user questions and generates responses using the conversation chain.
```
def main():
    # ... (UI setup)
    if user_question:
        if st.session_state.conversation:
            with st.spinner('Thinking very hard...'):
                try:
                    response = st.session_state.conversation.invoke({"question": user_question})
                    bot_response = response['answer']
                    st.session_state.chat_history.append((user_question, bot_response))
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        else:
            st.warning("Please process the documents first before asking questions.")
    # ... (display chat history)
```
***
### **Chat History Display and Document Upload**

The application uses Streamlit's layout features to create a main chat area and a sidebar for document uploads:
```
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
```

https://github.com/user-attachments/assets/7de0fdf5-9479-4f19-8558-1e084c421ea6

The chat history is displayed in the main area of the application:

- The ```chat_container``` is a Streamlit container that holds the chat messages.
- The chat history is stored in ```st.session_state.chat_history```.
- Messages are displayed in reverse order (most recent first) using the reversed() function.
- Each message is displayed using the ```display_message()``` function, which formats user and bot messages differently.
- A file uploader is provided for users to upload multiple PDF files.

When the "Process" button is clicked, the application:
- Extracts text from the PDFs using ```get_pdf_text()```.
- Splits the text into chunks using ```get_text_chunks()```.
- Creates a vector store from the chunks using ```get_vectorstore()```.
- Sets up the conversation chain using ```get_conversation_chain()```.
- If no PDFs are uploaded, a warning is displayed.


- A spinner is shown during document processing to indicate activity.

**End product looks like this:**

https://github.com/user-attachments/assets/63d7f8b7-d041-4e9c-a69c-54dd3fed3e31

***
I hope this that projects helps you understand the power of LLMs, ML, and Streamlit as well!
Since I was using Hugging Face API from their HuggingFaceHub, the size of the LLM was limited (Max 10GB, I believe, thus, I went with google/flan-t5-large) and amount of API calls is also limited. The good things is - this whole project is __100% free__. Most likely, you will get better results with OpenAI models or Anthropic models, but again, my option is __100% free__.

Follow me on:
- [LinkedIn](https://www.linkedin.com/in/iliakarelin/)
- [X](https://x.com/byiliakarelin)

Also, if you love ML, like this project and explanation of the code, consider subscribing to my [Newsletter](prosperindata.substack.com/subscribe).
