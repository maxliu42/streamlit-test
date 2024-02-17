# pip install streamlit langchain langchain-openai beautifulsoup4 python-dotenv chromadb pypdf2
# streamlit=1.13.1, langchain=0.1.6, langchain-community=0.0.19

import io
import streamlit as st
from PyPDF2 import PdfReader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# uses pdfreader to convert into text
def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = text_splitter = RecursiveCharacterTextSplitter()
    chunks = text_splitter.split_text(text)

    return chunks


def get_vectorstore(chunks):
    vector_store = Chroma.from_texts(chunks, OpenAIEmbeddings())

    return vector_store


# retrieves context for chat history
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain


# invoked for ai response
def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Generate a flash card style question and answer based on the below context. Give the user a single content-relevant question. Do not ask questions about titles or authors. After their guess, if it is correct give them another question. Only accept a user answer if it is the same as the actual answer. Otherwise, state their answer is incorrect. Provide a hint when asked. The hint must not contain the answer. Do not be overly nice and avoid apologizing. After providing an answer, always ask another question. \nThe text below is a json format from text that was scanned from a written note. So, it contains info about the locations of each piece of text. Just use that info when understanding it.\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    
    return response['answer']


def write_chat_history():
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)


# app config
st.set_page_config(page_title="Deerhackers", page_icon="🦌")
st.title("Deerhackers 🦌")

# user can upload files
with st.sidebar:
    st.subheader("Your documents")
    user_files = st.file_uploader(
        "Upload your files here. Allowed: *.txt, *.pdf", accept_multiple_files=True)

if not user_files:
    st.info("Please upload your files")

    # allows the user to change files by removing the old one
    if "vector_store" in st.session_state:
        st.session_state.pop("vector_store")
    if "chat_history" in st.session_state:
        st.session_state.pop("chat_history")


# once the user has uploaded their files
else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history =  [
                AIMessage(content="Are you ready to be grilled by OpenTA?"),
        ]

    # we parse the file the user uploaded into vectorstore for use in conversation
    if "vector_store" not in st.session_state:
        # we allow uploading pdf or text for now. if it's a pdf, we convert to text
        user_text = ""
        for user_file in user_files:
            if user_file.name.endswith(".pdf"):
                user_text += get_pdf_text(user_file)
            # we can use text directly
            elif user_file.name.endswith(".txt"):
                user_text += str(user_file.read())

            user_chunks = get_text_chunks(user_text)
            st.session_state.vector_store = get_vectorstore(user_chunks)
    

    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        # get response from ai and append
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))


    # write conversation to chat
    write_chat_history()