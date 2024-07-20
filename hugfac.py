import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from transformers import pipeline, HfApi

load_dotenv()
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")

# Set the API key for Hugging Face
if hf_api_key:
    HfApi().set_access_token(hf_api_key)

# Functions to process PDFs and build the conversational chain

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "please ask relevant questions from the pdf provided", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model_name = "deepset/roberta-base-squad2"
    qa_pipeline = pipeline("question-answering", model=model_name)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(qa_pipeline, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])

# Streamlit app

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF")

    user_question = st.text_input("Ask a Question from the PDF File/Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        # st.set_page_config(page_title="Chat PDF", page_icon=":books:")
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
