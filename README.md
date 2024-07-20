# Chat_with_PDF
Project Overview:
Developed a robust and interactive PDF chatbot capable of extracting information from PDF files and providing detailed answers to user queries.

Key Responsibilities:

Designed and implemented a PDF chatbot application using Streamlit for the front-end interface.
Utilized the LangChain framework to handle text processing, vector storage, and question-answering tasks.
Integrated Hugging Face's advanced language models for embeddings and question-answering functionalities.
Technologies Used:

Programming Languages: Python
Libraries and Frameworks: Streamlit, PyPDF2, LangChain, FAISS, Sentence-Transformers, Transformers
APIs: Hugging Face API
Tools: PyCharm for development, dotenv for environment variable management
Project Details:

PDF Processing: Extracted and processed text from multiple PDF files using PyPDF2.
Text Chunking: Split the extracted text into manageable chunks for efficient processing using LangChain's RecursiveCharacterTextSplitter.
Vector Store Creation: Created a vector store of text chunks using FAISS and embeddings from Hugging Face's sentence-transformers/all-mpnet-base-v2.
Question-Answering Chain: Developed a conversational chain to handle user queries, leveraging Hugging Face's deepset/roberta-base-squad2 model for accurate responses.
User Interface: Designed an intuitive interface using Streamlit, allowing users to upload PDF files and ask questions directly.
