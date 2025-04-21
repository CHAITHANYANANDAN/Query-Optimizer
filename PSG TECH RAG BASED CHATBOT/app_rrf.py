import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Qdrant  # ✅ Updated import
from langchain.embeddings import HuggingFaceEmbeddings  # ✅ New wrapper
from langchain_core.runnables import RunnableLambda
from langchain.schema.document import Document
from operator import itemgetter
from json import dumps, loads
import google.generativeai as genai
from qdrant_client import QdrantClient

# ---------------- CONFIGURATION ----------------

# Qdrant details
QDRANT_URL = "https://7296d5be-bc95-42ed-8a2c-48e86bd6009f.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.B1GAIaWf6_mFa_8Yk1oTpnngxwcvBddJyJ_eTr86eXo"
QDRANT_COLLECTION_NAME = "psg_dataset"

# Google Gemini API
GEMINI_API_KEY = "AIzaSyDvJbPbtrHaXsx5nS9eMQybYrY-vXJGJRU"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Retrieval parameters
TOP_K = 5
MAX_DOCS_FOR_CONTEXT = 5

# ---------------- INITIALIZATION ----------------

# Wrap SentenceTransformer with HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL, prefer_grpc=True, api_key=QDRANT_API_KEY)


# ---------------- RRF IMPLEMENTATION ----------------

def reciprocal_rank_fusion(results: list[list], k=60) -> list[Document]:
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps({"page_content": doc.page_content, "metadata": doc.metadata})
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return [Document(**x[0]) for x in reranked_results[:MAX_DOCS_FOR_CONTEXT]]


def rrf_retriever(query: str) -> list[Document]:
    qdrant = Qdrant(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION_NAME,
        embeddings=embedding_model,
    )

    retriever1 = qdrant.as_retriever(search_kwargs={"k": TOP_K})
    retriever2 = qdrant.as_retriever(search_kwargs={"k": TOP_K})

    def chain(query):
        docs1 = retriever1.invoke(query)  # ✅ Replaced get_relevant_documents
        docs2 = retriever2.invoke(query)
        return reciprocal_rank_fusion([docs1, docs2])

    return chain(query)


# ---------------- GEMINI GENERATION ----------------

def gemini(query, chunks):
    context = "\n".join([f"{i+1}. {chunk}" for i, chunk in enumerate(chunks)])
    prompt = f"""
You are a helpful and knowledgeable college information assistant. Using the provided context from the college website, answer the user's query clearly and accurately.

### Context:
{context}

### Query:
{query}

Based on the above context, provide a relevant and helpful answer to the user's question about the college. Keep the response concise, clear, and informative.
"""
    response = model.generate_content(prompt)

    if response.candidates and len(response.candidates) > 0:
        return response.text
    else:
        return "No valid content was returned. Please adjust your query or try again."


# ---------------- MAIN LOGIC ----------------

def getResult(input_query):
    context_docs = rrf_retriever(input_query)
    chunks = [doc.page_content for doc in context_docs]
    return gemini(input_query, chunks)


# ---------------- STREAMLIT UI ----------------

st.set_page_config(
    page_title="PSG Tech Info Search",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown(
    "<h2 style='text-align: center;'>🎓 PSG College of Technology Info Search</h2>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Ask anything about PSG Tech and get instant smart results.</p>", 
    unsafe_allow_html=True
)

st.markdown("###")

query = st.text_input("🔍 Enter your query", placeholder="e.g., Who is the Dean of Placements?")
search = st.button("Search")

if search:
    if query.strip():
        with st.spinner("Searching and generating results..."):
            result = getResult(query)

        st.markdown("---")
        st.markdown("#### 📘 Answer:")
        st.markdown(result)
    else:
        st.warning("⚠️ Please enter a valid query.")
