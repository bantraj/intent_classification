import os
import pickle
from datetime import datetime
from enum import Enum
from pathlib import Path
import logging
import io
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import time

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("ingestion_service")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()
app = FastAPI()
CHROMA_PATH = "./chroma_db"
BM25_PATH = "./bm25_indices"

Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
Path(BM25_PATH).mkdir(parents=True, exist_ok=True)


class SearchType(str, Enum):
    COSINE = "Cosine Similarity"
    HYBRID = "Hybrid RAG"
    BM25 = "BM25"


@app.post("/ingestion")
async def ingestion(file: UploadFile = File(...)):
    logger.info("ingestion request received filename=%r content_type=%r", file.filename, file.content_type)

    if not file.filename.endswith((".xls", ".xlsx")):
        logger.warning("unsupported file extension filename=%r", file.filename)
        raise HTTPException(status_code=400, detail="File extension not supported")
    try:
        contents = await file.read()  # Await the coroutine to get the actual bytes
        buffer = io.BytesIO(contents)  # Wrap bytes in a buffer Pandas can read
        df = pd.read_excel(buffer)
        logger.info("excel loaded rows=%d cols=%d", df.shape[0], df.shape[1])
        docs = []
        for _, row in df.iterrows():
            content = f"Utterance: {row['Utterance']}"
            metadata = {"intent": row['Intent'], "original_utterance": row['Utterance']}
            docs.append(Document(page_content=content, metadata=metadata))

        logger.info("documents prepared count=%d", len(docs))

            # Generate Collection Name
        timestamp = datetime.now().strftime("%d%m%y%H%M")
        clean_name = "".join(filter(str.isalnum, file.filename.split('.')[0]))
        collection_name = f"{clean_name}_{timestamp}"
        logger.info("collection_name generated %r", collection_name)

        Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=CHROMA_PATH
        )
        logger.info("chroma persisted collection=%r dir=%r", collection_name, CHROMA_PATH)

        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_file = f"{BM25_PATH}/{collection_name}.pkl"
        with open(f"{BM25_PATH}/{collection_name}.pkl", "wb") as f:
            pickle.dump(bm25_retriever, f)

        logger.info("bm25 persisted path=%r", bm25_file)

        return {"status": "success", "collection_name": collection_name}
    except Exception as e:
        logger.exception("ingestion failed filename=%r", file.filename)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/search")
async def search_intent(collection_name: str, query: str, mode: SearchType):
    # Helper to load BM25
    logger.info(
        "search request collection=%r mode=%r query_len=%d",
        collection_name,
        mode,
        len(query or ""),
    )
    t0 = time.perf_counter()
    def load_bm25():
        file_path = f"{BM25_PATH}/{collection_name}.pkl"
        if not os.path.exists(file_path):
            logger.warning("bm25 index not found path=%r", file_path)
            raise HTTPException(status_code=404, detail="BM25 index not found.")
        with open(file_path, "rb") as f:
            retriever = pickle.load(f)
            retriever.k = 1  # Ensure top 1
            logger.info("bm25 loaded path=%r k=%d", file_path, retriever.k)
            return retriever

    # 1. Load the Chroma Vector Store
    vector_db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    logger.info("chroma loaded collection=%r dir=%r", collection_name, CHROMA_PATH)
    vs_retriever = vector_db.as_retriever(search_kwargs={"k": 1})

    if mode == SearchType.COSINE:
        retriever = vs_retriever
        logger.info("retriever selected=vector k=1")
    elif mode == SearchType.BM25:
        retriever = load_bm25()
        logger.info("retriever selected=bm25")
    elif mode == SearchType.HYBRID:
        bm_retriever = load_bm25()
        retriever = EnsembleRetriever(
            retrievers=[bm_retriever, vs_retriever],
            weights=[0.5, 0.5]
        )
        logger.info("retriever selected=hybrid weights=[0.5,0.5]")

    results = retriever.invoke(query)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)  # <-- end timing
    logger.info(
        "retrieval completed results_count=%d latency_ms=%.2f",
        len(results or []),
        elapsed_ms,
    )

    if not results:
        return {"message": "No match found", "latency_ms": elapsed_ms}

    return {
        "search_mode": mode,
        "matched_utterance": results[0].page_content,
        "detected_intent": results[0].metadata.get("intent")
    }

#
# if __name__ == "__main__":
#     import uvicorn
# 
#     uvicorn.run(app, host="0.0.0.0", port=8000)
