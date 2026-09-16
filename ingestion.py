from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_unstructured import UnstructuredLoader
# from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings


load_dotenv()

# 1. Scrape the URLs and load content into documents.
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


docs = [UnstructuredLoader(web_url=url, chunking_strategy="basic", max_characters=1000000).load() for url in urls]

docs_list = [item for sublist in docs for item in sublist]


# 2. Convert loaded documents into chunks.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)


# 3. Index the content into ChromaDB.
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

# IMPORTANT: Note that this logic will run only once to index the content, after that you should comment it.
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding=embeddings,
#     persist_directory="./.chroma",
# )

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=embeddings,
).as_retriever()


