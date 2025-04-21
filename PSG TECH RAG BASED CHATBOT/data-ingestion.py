from langchain.text_splitter import RecursiveCharacterTextSplitter


from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer  # Example embedding model (open-source)
import uuid
from langchain_core.documents.base import Document
import pathlib

# read pdf files and chunk them into pieces
# upload chunks to Qdrant


QDRANT_URL = "https://7296d5be-bc95-42ed-8a2c-48e86bd6009f.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.B1GAIaWf6_mFa_8Yk1oTpnngxwcvBddJyJ_eTr86eXo" # Use your Qdrant API key
#QDRANT_COLLECTION_NAME = "dataset"

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL, prefer_grpc=True, api_key=QDRANT_API_KEY)

def create_QDrant_collection(collectionName):
    """Create Qdrant collection."""
    
    # Define collection parameters
    collection_name = collectionName
    vector_size = 384  # Size of the embedding vectors
    distance_metric = models.Distance.COSINE  # Distance metric for vector similarity

    # Check if the collection already exists
    if qdrant_client.collection_exists(collection_name):
        # Optionally, delete the existing collection if you need to recreate it
        qdrant_client.delete_collection(collection_name)

    # Create the collection with the new method
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=distance_metric)  # Pass as vectors_config
    )

    print(f"Collection '{collection_name}' created successfully.")


def read_txt_files(directory: str) -> list[Document]:
    """Reads all .txt files in a given directory and returns a list of Document objects.

    Args:
        directory (str): The path to the directory containing .txt files.

    Returns:
        list[Document]: A list of Document objects containing the content of the .txt files.
    """
    documents = []
    for txt_file in pathlib.Path(directory).glob('*.txt'):
        with open(txt_file, 'r',encoding="utf-8") as file:
            content = file.read()
            metadata = {"filename": txt_file.stem}
            documents.append(Document(page_content=content, metadata=metadata))
    return documents



def upload_chunks_to_QDrant(documents, collectionName):
    records_to_upload = []
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 

    for idx, chunk in enumerate(documents):
        content = chunk.page_content
        filename = chunk.metadata.get("filename", "unknown.txt")  # Get filename from metadata
        
        # Get the embedding for the content
        vector = embedding_model.encode(content).tolist()  # Use encode method for getting the vector

        record = models.PointStruct(
            id=idx,
            vector=vector,
            payload={"page_content": content, "filename": filename}  # Store filename as part of payload
        )
        records_to_upload.append(record)

    qdrant_client.upload_points(
        collection_name=collectionName,
        points=records_to_upload
    )

    return

# Example usage

#def collection_create(pdf_path,collection_name):
#     print("collection create called")
#     create_QDrant_collection(collection_name)
#     pdf_file_path = pdf_path
#     chunks= chunk_pdf_text(pdf_file_path)
#     upload_chunks_to_qdrant(chunks,collection_name)

#create_QDrant_collection("psg_dataset")
documents = read_txt_files("A:\PROJECTS\COLLEGE PACKAGES\RAG BASED QUERY SEARCH OPTIMISER FOR PSG TECH WEBSITE\Data set")
upload_chunks_to_QDrant(documents, "psg_dataset")