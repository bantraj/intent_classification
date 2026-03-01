import streamlit as st
import requests

# Set page config
st.set_page_config(page_title="Intent Search Engine", layout="wide")

st.title("🚀 Intent AI: Excel to Vector Search")
st.markdown("Upload your Excel file to create embeddings, then query them using different search algorithms.")

# Sidebar for API Configuration
st.sidebar.header("Settings")
api_url = st.sidebar.text_input("FastAPI URL", value="http://localhost:8000")

# --- STEP 1: UPLOAD ---
st.header("1. Data Ingestion")
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if st.button("Upload and Process"):
    if uploaded_file is not None:
        with st.spinner("Processing embeddings... this may take a moment."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                response = requests.post(f"{api_url}/ingestion/", files=files)
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Success! Collection Created: {data['collection_name']}")
                    st.session_state['collection_name'] = data['collection_name']
                else:
                    st.error(f"Error: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Connection failed: {e}")
    else:
        st.warning("Please upload a file first.")

st.divider()

# --- STEP 2: SEARCH ---
st.header("2. Search & Retrieval")

# Check if we have a collection name in session or input it manually
col_name = st.text_input("Active Collection Name",
                         value=st.session_state.get('collection_name', ""))

query = st.text_input("Enter User Query (e.g., 'How do I reset my password?')")

search_mode = st.selectbox(
    "Select Search Type",
    ["Cosine Similarity", "Hybrid RAG", "BM25"]
)

if st.button("Search Intent"):
    if col_name and query:
        with st.spinner("Searching..."):
            params = {
                "collection_name": col_name,
                "query": query,
                "mode": search_mode
            }
            try:
                # Note: Using GET as defined in your previous FastAPI snippet
                response = requests.get(f"{api_url}/search/", params=params)
                if response.status_code == 200:
                    result = response.json()

                    # Display Results in nice columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Detected Intent", result['detected_intent'])
                    with col2:
                        st.info(f"Matched Utterance: {result['matched_utterance']}")
                else:
                    st.error(f"Search failed: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Search error: {e}")
    else:
        st.warning("Ensure Collection Name and Query are provided.")