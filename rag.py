import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()


## LOAD DATA

def load_documents(docs_data):
    if not  os.path.exists(docs_data):
        raise FileNotFoundError("the data file was not found")

    loader = DirectoryLoader(
        path = docs_data,
        glob = "*.txt",
        loader_cls = TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()
    if len(documents) == 0:
        raise FileNotFoundError(f"no .txt files in {docs_data}")
    for i,doc in enumerate(documents[:2]):
        print(f"document {i+1}  :")
        print(doc.metadata['source'])
    return documents





def chunking(documents,chunk_size = 800, chunk_overlap = 0):
    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    for i,chunk in enumerate(chunks[:5]):
        print(f"--- chunk {i+1} ---")
        print(f"source : {chunk.metadata['source']}")
        print(f"length : {len(chunk.page_content)} characters")
        print(chunk.page_content)
    return chunks


def vector_store(chunks,dir = "db/chroma_db"):
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    print("--- creating vector database ---")
    vector_store = Chroma(
        persist_directory=dir,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space":"cosine"}
    )

    vector_store.add_documents(chunks)
    print("--- finished creating vector database ---")

    return vector_store


def main():
    documents = load_documents(docs_data="docs")
    chunks = chunking(documents=documents)
    vector_store(chunks=chunks)

    




if __name__ == "__main__":
    main()