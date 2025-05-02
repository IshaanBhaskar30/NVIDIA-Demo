📊 US Census Q&A Chatbot using NVIDIA NIM + FAISS

This project demonstrates a document-based question answering (RAG) chatbot powered by NVIDIA NIM's LLaMA 3.1 70B Instruct model and FAISS vector search, integrated into a simple and interactive Streamlit application. The app allows users to upload or query preloaded US Census PDF documents—such as data on poverty rates, income, occupation, and health insurance coverage—and receive fast, accurate, and context-aware answers based on those documents.

Key features:

->⚡ NVIDIA ChatNVIDIA LLM (LLaMA 3.1 70B) for powerful natural language understanding

->🔍 NVIDIAEmbeddings for document vectorization using state-of-the-art models

->🧠 RAG pipeline combining retrieval and generation using LangChain’s high-level chains

->🗂️ FAISS vector database to store and retrieve semantically similar chunks

->📄 Sample documents include:

    o Health Insurance Coverage Status by Geography (2021–2022)

    o Poverty in States and Metropolitan Areas (2022)

    o Household Income in States and Metro Areas (2022)

    o Occupation, Earnings, and Job Characteristics

Ideal for:

->Data analysts, policy researchers, or journalists looking to interactively explore U.S. Census insights

->Anyone wanting to query large documents conversationally using cutting-edge NVIDIA AI APIs

The system performs all chunking, embedding, vector indexing, and semantic retrieval in real time—delivering answers in seconds through a simple UI.
