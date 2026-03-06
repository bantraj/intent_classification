import streamlit as st
import requests

# Set page config for a professional look
st.set_page_config(
    page_title="Intent AI: Query Improviser",
    page_icon="🪄",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Intent AI: Excel to Vector Search")
st.markdown("Upload your Excel file to create embeddings, then use the LLM to improvise and rephrase queries.")

# Sidebar for API Configuration
st.sidebar.header("⚙️ Configuration")
api_url = st.sidebar.text_input("FastAPI URL", value="http://localhost:8000")
st.sidebar.info("Ensure your FastAPI backend is running at the address above.")

# --- STEP 1: DATA INGESTION ---
st.header("1️⃣ Data Ingestion")
with st.container():
    uploaded_file = st.file_uploader("Choose an Excel file (Intent/Utterance format)", type=["xlsx", "xls"])

    if st.button("🚢 Upload and Process"):
        if uploaded_file is not None:
            with st.spinner("Processing embeddings... this may take a moment."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                try:
                    # Pointing to your @app.post("/ingestion") endpoint
                    response = requests.post(f"{api_url}/ingestion", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Success! Collection Created: {data['collection_name']}")
                        st.session_state['collection_name'] = data['collection_name']
                    else:
                        st.error(f"❌ Error: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"⚠️ Connection failed: {e}")
        else:
            st.warning("Please upload a file first.")

st.divider()

# --- STEP 2: SEARCH & IMPROVISATION ---
st.header("2️⃣ Search & Improvisation")

# Layout for inputs
col_input1, col_input2 = st.columns([2, 1])

with col_input1:
    query = st.text_input("Enter User Query", placeholder="e.g., How do I gain access to a building?")

with col_input2:
    search_mode = st.selectbox(
        "Search Type",
        ["Hybrid RAG", "Cosine Similarity", "BM25"]
    )

# Collection name persistence
col_name = st.text_input("Active Collection Name",
                         value=st.session_state.get('collection_name', ""),
                         help="This is automatically filled after upload.")

if st.button("🔍 Analyze & Rephrase"):
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

                    # --- Display Core Classification Results ---
                    st.subheader("🎯 Classification Results")
                    res_col1, res_col2 = st.columns([1, 2])

                    with res_col1:
                        st.metric("Detected Intent", result.get('detected_intent', 'N/A'))

                    with res_col2:
                        st.info(f"**Matched Utterance (DB):**\n\n{result.get('matched_utterance', 'N/A')}")

                    st.divider()

                    # --- Handle Nested JSON for Rephrased Queries ---
                    st.subheader("💡 Suggested Rephrasings")
                    st.write("The LLM has generated these improvisations based on the matched intent:")

                    # Access the nested list: result['queries'] -> [{'query': '...'}]
                    queries_list = result.get('queries', [])

                    if queries_list:
                        # Displaying variations in a grid
                        cols = st.columns(2)
                        for idx, item in enumerate(queries_list):
                            # Extracting the string from the 'query' key in each dictionary
                            rephrased_text = item.get("query", "No text found")

                            with cols[idx % 2]:
                                st.success(f"**Variation {idx + 1}:**\n{rephrased_text}")
                    else:
                        st.warning("The intent was matched, but no rephrased queries were returned by the LLM.")

                    # Latency tracking
                    if "latency_ms" in result:
                        st.caption(f"Backend processing time: {result['latency_ms']} ms")

                else:
                    st.error(f"Search failed: {response.json().get('detail', 'Unknown Error')}")
            except Exception as e:
                st.error(f"UI Error: {str(e)}")
    else:
        st.warning("Please ensure both Collection Name and Query are provided.")

# Sidebar Footer
st.sidebar.divider()
if st.sidebar.button("🗑️ Clear Collection History"):
    st.session_state['collection_name'] = ""
    st.rerun()