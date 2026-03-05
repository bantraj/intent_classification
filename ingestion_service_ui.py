import streamlit as st
import requests

# Set page config
st.set_page_config(page_title="Intent Search Engine", layout="wide", page_icon="🪄")

st.title("🚀 Intent AI: Query Improviser")
st.markdown(
    "Upload your Excel file to create embeddings, then see how the AI improvises your queries based on matched intents.")

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
                # Fixed endpoint path to match your FastAPI @app.post("/ingestion")
                response = requests.post(f"{api_url}/ingestion", files=files)
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

# --- STEP 2: SEARCH & IMPROVISE ---
st.header("2. Search & Improvisation")

col_name = st.text_input("Active Collection Name",
                         value=st.session_state.get('collection_name', ""))

query = st.text_input("Enter User Query (e.g., 'How do I gain access to a building?')")

search_mode = st.selectbox(
    "Select Search Type",
    ["Hybrid RAG", "Cosine Similarity", "BM25"]
)

if st.button("Analyze & Rephrase"):
    if col_name and query:
        with st.spinner("Analyzing intent and rephrasing..."):
            params = {
                "collection_name": col_name,
                "query": query,
                "mode": search_mode
            }
            try:
                # API Call to updated backend
                response = requests.get(f"{api_url}/search", params=params)

                if response.status_code == 200:
                    result = response.json()

                    # Display Core Results
                    st.subheader("🎯 Classification Results")
                    c1, c2, c3 = st.columns([2, 3, 1])
                    with c1:
                        st.metric("Detected Intent", result['detected_intent'])
                    with c2:
                        st.info(f"**Matched Source:** {result['matched_utterance']}")
                    with c3:
                        st.write(f"**Latency:** {result.get('latency_ms', 0)}ms")

                    st.divider()

                    # NEW: Display Rephrased Queries
                    st.subheader("💡 Suggested Rephrasings")
                    st.write("The LLM generated these variations to help you expand your dataset or verify the intent:")

                    rephrased_list = result.get('rephrased_queries', [])

                    if rephrased_list:
                        # Displaying as clean, numbered info boxes
                        cols = st.columns(2)  # Split into two columns for better space usage
                        for idx, q in enumerate(rephrased_list):
                            with cols[idx % 2]:
                                st.success(f"**Variation {idx + 1}:** \n{q}")
                    else:
                        st.warning("No rephrasings were generated for this query.")

                else:
                    st.error(f"Search failed: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Search error: {e}")
    else:
        st.warning("Ensure Collection Name and Query are provided.")