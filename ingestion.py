import os

from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

if __name__ == "__main__":

    # 1. Initialize Ollama local embeddings
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    # 2. Create sample documents
    print("Loading document...")

    loader = UnstructuredLoader(file_path="./mediumblog1.txt", chunking_strategy="basic", max_characters=1000000)
    document = loader.load()

    # 3. Splitting document.
    print("splitting...")
    # Note that chunk size should be specified carefully to fit the context window of the used LLM. So if the maximum context window for the used LLM let's say 150,000 cunk size shouldn't be exceed that number.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document) # It will return a list of Document object, each document content has chunk_size(1000) character.
 
    print(f"created {len(texts)} chunks")

    # 4. Splitting document.
    print("ingesting...")

    Chroma.from_documents(
        documents=document, embedding=embeddings, persist_directory="./chroma_db"
    )

    print("finish")
