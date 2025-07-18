## The name LangChain comes from:

- Lang = Language (as in Language Models like GPT)
- Chain = Linking together multiple steps or tools in a pipeline

## LangChain lets you chain together:

- LLMs (like GPT, Claude, Gemini)
- Tools (search engines, calculators, APIs)
- Documents (PDFs, databases, websites)
- Logic and memory (previous answers, reasoning steps)

## pypdf

-connect python with pdf

## Files:

- basic_app.py: Has flask implementation basic
- document_processor.py : Has basic document storing and processing using PyPDFLoader and langchain
- vector-store.py: Has basic chroma db use to store embeddings and calculate similarity of embeddings to answer our question from pdf

# Difference of using LLM to generate answer

It tries to create more human-like and make it more elaborate by the use of LLM rather than just using the text from the pdf
