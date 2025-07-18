from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

#initialize components
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
persist_directory="chroma_db"

#Process pdf and store in vector database
def process_pdf_and_store_file(file_path):
    loader=PyPDFLoader(file_path)
    documents=loader.load()
    chunks=text_splitter.split_documents(documents)
    vector_store=Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vector_store.persist()
    return vector_store

#Test similarity search
def test_search(vector_store,query):
    results=vector_store.similarity_search(query,k=3)
    for i,doc in enumerate(results):
        print(f"Result {i+1}: {doc.page_content[:200]}..")

if __name__=="__main__":
    pdf_path='Flask_documentation.pdf'
    if not os.path.exists(pdf_path):
        print('Please provide pdf path')
    else:
        os.makedirs(persist_directory,exist_ok=True)
        vector_store=process_pdf_and_store_file(pdf_path)
        test_query="What is the main topic of document?"
        test_search(vector_store, test_query)