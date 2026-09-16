import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Load the txt documents
def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory '{docs_path}' does not exist.")

    # Load all txt files from the  docs directory
    loader = DirectoryLoader(docs_path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(f"No .txt found in the directory '{docs_path}'.")

    # Show the first 2 documents 
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1}:")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:200]}...")  # Print first 200 characters
        print(f" Metadata: {doc.metadata}")
    
    return documents


    # Chunking the documents
def split_documents(documents, chunk_size=800, chunk_overlap=100):
    print(
        f"\nSplitting documents into chunks of size "
        f"{chunk_size} with overlap {chunk_overlap}..."
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks[:5]):
        print(f"\nChunk {i + 1}:")
        print(f"Content length: {len(chunk.page_content)} characters")
        print(f"Content preview: {chunk.page_content[:200]}...")
        print(f"Metadata: {chunk.metadata}")

    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks.")

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("---- Creating vector store in directory -----")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma.from_documents(documents=chunks,embedding=embedding_model, persist_directory=persist_directory, collection_metadata={"hnsw:space": "cosine"})
    print("---- Finished creating vector store in directory -----")
    print(f"Vector store created and saved in directory: {persist_directory}")
    return vector_store

def main():
    print("Main function")

    # Load the documents
    documents = load_documents(docs_path="docs")

    # Split the documents into chunks
    chunks = split_documents(documents)

    # Create the vector store
    vectorstore = create_vector_store(chunks)


if __name__ == "__main__":
    main()