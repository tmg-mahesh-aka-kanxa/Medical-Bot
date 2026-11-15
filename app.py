import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_resources():
    index = faiss.read_index("medical_index.faiss")
    df = pd.read_pickle("medical_data.pkl")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, df, model

index, df, model = load_resources()

st.title("Medical Q&A Assistant")
st.caption("For informational purposes only, not medical advice.")

if query := st.chat_input("Ask a general medical question..."):
    st.chat_message("user").write(query)

    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    similarities, indices = index.search(query_embedding, k=1)

    if similarities < 0.45:
        response = "I'm sorry, I don't have enough information to answer that question safely."
    else:
        matched_q = df.iloc[indices, 0]
        answer = df.iloc[indices, 1]
        response = f"**Matched Topic:** {matched_q}\n\n{answer}"

    st.chat_message("assistant").write(response)
