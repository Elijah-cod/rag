import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
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

def main():
    print("Main function")



if __name__ == "__main__":
    main()