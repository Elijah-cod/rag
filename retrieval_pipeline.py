from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
# Access the environment variables from the .env file
load_dotenv()

# Load the storage directory
persist_directory = "db/chroma_db"

# Load embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize the Chroma vector store
db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model, collection_metadata={"hnsw:space": "cosine"})

query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(
    search_kwargs={"k": 10}
)

relevant_docs = retriever.invoke(query)

print("--- Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"\nDocument {i}:")
    print(doc.page_content)
    print(f"Source: {doc.metadata.get('source')}")