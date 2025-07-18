from langchain.document_loader import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
import os

#initialize components
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
persist_directory="chroma_db"
MODEL_NAME="distilgpt2"

#initialize LLM
def initialize_llm():
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
    model=AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    pipe=pipeline("text-generation",model=model,tokenizer=tokenizer,max_length=512)
    return HuggingFacePipeline(pipeline=pipe)

#Process pdf and create QA chain
#Loads the PDF → splits it → embeds chunks → stores in Chroma.
#Sets up a RetrievalQA chain that:
#--Retrieves top 3 similar chunks
#--Passes them as context to the LLM
#--LLM tries to generate a direct answer

def create_qa_chain(file_path):
    loader=PyPDFLoader(file_path)
    documents=loader.load()
    chunks=text_splitter.split_documents(documents)
    vector_store=Chroma.from_documents(
        documents=chunks,
        embeddings=embeddings,
        persist_directory=persist_directory
    )
    vector_store.persist()
    llm=initialize_llm()
    retriever=vector_store.as_retriever(search_kwargs={"k":3})
    qa_chain=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

#Test QA chain
if __name__=="__main__":
    pdf_path="Flask_documentation.pdf"
    if not os.path.exists(pdf_path):
        print("Please provide pdf path")
    else:
        qa_chain=create_qa_chain(pdf_path)
        query="What is the main topic of document?"
        result=qa_chain({"query":query})
        print(f"Response:{result[result]}")
        print("\n Sources")
        for i,doc in enumerate(result['source_documents']):
            print(f"Source {i+1}: {doc.page_content[:200]}..") 