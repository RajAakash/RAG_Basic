from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 

#initialize text splitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

#Load and process pdf
def process_pdf(file_path):
    loader=PyPDFLoader(file_path)
    documents=loader.load()
    chunks=text_splitter.split_documents(documents)
    return chunks

#Save chunks to file
def save_chunks(chunks, outputs_file):
    with open(outputs_file,"w",encoding='utf-8') as f:
        for i,chunk in enumerate(chunks):
            f.write(f"Chunk {i+1}: \n {chunk.page_content} \n\n")

if __name__=="__main__":
    pdf_path="Flask_documentation.pdf"
    if not os.path.exists(pdf_path):
        print("Provide a valid pdf path")
    else:
        chunks=process_pdf(pdf_path)
        save_chunks(chunks,"chunks.txt")
        print(f"Processed {chunks} and saved to chunks.txt")
