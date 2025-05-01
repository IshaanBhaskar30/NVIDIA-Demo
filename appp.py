import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

# Page title
st.title("Nvidia NIM Demo")

# Prompt user for NVIDIA API key
nvidia_api_key = st.text_input("🔐 Enter your NVIDIA API Key:", type="password")

if nvidia_api_key:
    os.environ["NVIDIA_API_KEY"] = nvidia_api_key

    # Load the LLM
    llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")

    def vector_embedding():
        if 'vectors' not in st.session_state:
            st.session_state.embeddings = NVIDIAEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Data Ingestion
            st.session_state.docs = st.session_state.loader.load()  # Document Loading
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunking
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs[:30])  # Splitting
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents, st.session_state.embeddings)  # Vector DB

    prompt_template = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Question: {input}
        """
    )

    user_question = st.text_input("💬 Enter your question from the documents:")

    if st.button("📄 Create Document Embeddings"):
        vector_embedding()
        st.success("✅ FAISS Vector Store DB is ready using NVIDIAEmbeddings")

    if user_question:
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_question})
        elapsed_time = time.process_time() - start

        st.markdown(f"⏱️ **Response Time:** {elapsed_time:.2f} seconds")
        st.write("🧠 **Answer:**", response['answer'])

        with st.expander("🔍 Document Similarity Search Results"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("---")

else:
    st.warning("Please enter your NVIDIA API key to continue.")

