import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import fitz
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(documents):
    text = ""
    for document in documents:
        if hasattr(document, 'metadata') and isinstance(document.metadata, dict):
            pdf_file_name = document.metadata.get('source')
            if pdf_file_name:
                doc = fitz.open(pdf_file_name)
                for page in doc:
                    text += page.get_text()
    return text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(query):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(query)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": query}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    loader = PyPDFLoader("C:/Users/AGL/OneDrive/Ash_New proj/PDF Reader/Gemini/NLP interview.pdf")
    documents = loader.load()
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")

    query = st.text_input("Ask a Question from the uploaded PDF File")
    
    if query:
        user_input(query)

    raw_text = get_pdf_text(documents)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
            


if __name__ == "__main__":
    main()