import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="BioMedical AI Search", page_icon="🧬", layout="centered")

st.title("🧬 Medical Literature Semantic Search")
st.markdown("""
Welcome to the Clinical Discovery Engine. This tool searches through clinical text using a
**hybrid PubMedBERT + BM25 engine**, attempting blending semantic understanding with exact keyword matching.
""")


@st.cache_resource
def load_model():
    return SentenceTransformer('NeuML/pubmedbert-base-embeddings')


@st.cache_resource
def load_bm25():
    bm25 = BM25Encoder()
    bm25.load("bm25_model.json")
    return bm25


@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    return pc.Index("medical-hybrid-search")


def hybrid_scale(dense: list, sparse: dict, alpha: float) -> tuple:
    """
    Scale dense and sparse vectors by alpha.
    alpha=1.0 -> pure semantic  |  alpha=0.0 -> pure keyword
    """
    hsparse = {
        "indices": sparse["indices"],
        "values": [v * (1 - alpha) for v in sparse["values"]],
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse


with st.spinner("Loading AI models and vector database..."):
    model = load_model()
    bm25 = load_bm25()
    index = init_pinecone()

st.write("---")

query = st.text_input(
    "Enter clinical symptoms or a research topic:",
    placeholder="e.g., patient presenting with pulmonary nodules and severe coughing..."
)

# Alpha slider — the core Phase 2 UI addition.
# Left = pure keyword (BM25), Right = pure semantic (PubMedBERT).
# Default 0.75: semantic-dominant but BM25 helps with exact medical terms.
alpha = st.slider(
    "Search mode",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05,
    help="Semantic: understands meaning and synonyms. Keyword: exact term matching. "
         "Hybrid blends both — useful when your query contains specific gene names, "
         "drug names, or technical identifiers."
)

col1, col2 = st.columns(2)
col1.caption(f"← Pure keyword (BM25)  |  alpha = {alpha:.2f}  |  Pure semantic (PubMedBERT) →")

if st.button("Search Literature", type="primary"):
    if query:
        with st.spinner("Searching vector database..."):
            dense_vec = model.encode(query, show_progress_bar=False).tolist()
            sparse_vec = bm25.encode_queries(query)
            hdense, hsparse = hybrid_scale(dense_vec, sparse_vec, alpha)

            results = index.query(
                vector=hdense,
                sparse_vector=hsparse,
                top_k=5,
                include_metadata=True,
            )

        st.subheader("Top Search Results")
        matches = results.get("matches", [])

        if not matches:
            st.warning("No results found. Try adjusting the search mode slider or rephrasing your query.")
        else:
            for match in matches:
                details = match.get("metadata", {})
                diagnosis = details.get("diagnosis", "Unknown")
                snippet = details.get("text_snippet", "No snippet available.")
                score = match["score"]

                with st.container():
                    st.markdown(f"#### Classification: `{diagnosis}`")
                    st.caption(f"Hybrid Match Score: {score:.4f}  |  alpha = {alpha:.2f}")
                    st.info(snippet)
                    st.divider()
    else:
        st.warning("Please enter a query to search.")
