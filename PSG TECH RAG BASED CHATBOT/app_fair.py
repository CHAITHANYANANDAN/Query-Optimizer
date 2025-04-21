import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import google.generativeai as genai
import os
import base64

# Qdrant details
QDRANT_URL = "https://7296d5be-bc95-42ed-8a2c-48e86bd6009f.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.B1GAIaWf6_mFa_8Yk1oTpnngxwcvBddJyJ_eTr86eXo" # Use your Qdrant API key
QDRANT_COLLECTION_NAME = "psg_dataset"

# Google Gemini API details
GEMINI_API_KEY = "AIzaSyDvJbPbtrHaXsx5nS9eMQybYrY-vXJGJRU"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL, prefer_grpc=True, api_key=QDRANT_API_KEY)

# Load the SentenceTransformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Vector size = 384


def vector_search(query, collection_name, top_k):
    """Perform a vector search on the Qdrant collection."""
    query_vector = embedder.encode(query).tolist()
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    results = []
    for result in search_result:
        chunk_text = result.payload.get('page_content', 'No text found')
        results.append(chunk_text)
    return results


def gemini(query, chunks):
    """Generates an answer using Google's Generative AI (Gemini)."""
    context = "\n".join([f"{i+1}. {chunk}" for i, chunk in enumerate(chunks)])
    prompt = f"""
    You are a helpful and knowledgeable college information assistant. Using the provided context from the college website, answer the user's query clearly and accurately.

    
    ### Context:
    {context}
    
    ### Query:
    {query}
    
    Based on the above context, provide a relevant and helpful answer to the user's question about the college.
    Provide a concise, clear, and informative response based on the query.
    """
    # Make the request to generate text
    response = model.generate_content(prompt)

    # Check if the response contains valid content
    if response.candidates and len(response.candidates) > 0:
        return response.text    # Return the generated text as a string
    else:
        return "No valid content was returned. Please adjust your prompt or try again."


def getResult(input_query):
    context = vector_search(input_query, QDRANT_COLLECTION_NAME, top_k=5)
    return gemini(input_query, context)




#Streamlit App
st.title("PSG COLLEGE OF TECHNOLOGY")
st.write("Search for any information about PSG COLLEGE OF TECHNOLOGY using a query. The system retrieves and generates relevant details.")

# Search bar for user input
query = st.text_input("Enter your query:", "")

# Display the result when the user enters a query
if st.button("Search"):
    if query.strip():
        with st.spinner("Searching and generating results..."):
            result = getResult(query)
        st.subheader("Results:")
        st.write(result)
    else:
        st.warning("Please enter a valid query!")
