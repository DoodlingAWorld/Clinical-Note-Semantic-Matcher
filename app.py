import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

# Configure the Streamlit page
st.set_page_config(page_title="BioMedical AI Search", page_icon="🧬", layout="centered")

st.title("🧬 Medical Literature Semantic Search")
st.markdown("""
Welcome to the Clinical Discovery Engine. This tool searches through clinical text using a **weighted PubMedBERT vector engine** to prioritize diagnoses and abstract summaries.
""")


# Caches Model Loading for performance
@st.cache_resource
def load_model():
    # Streamlit will only run this once!
    return SentenceTransformer('NeuML/pubmedbert-base-embeddings')


# Cache the Pinecone Connection
@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    return pc.Index("medical-weighted-search")


# Load resources
with st.spinner("Loading AI Model and Vector Database..."):
    model = load_model()
    index = init_pinecone()

# Build the User Interface
st.write("---")
query = st.text_input(
    "Enter clinical symptoms or a research topic:",
    placeholder="e.g., patient presenting with pulmonary nodules and severe coughing..."
)

# Execute the Search
if st.button("Search Literature", type="primary"):
    if query:
        with st.spinner("Searching vector database..."):
            query_embedding = model.encode(query, show_progress_bar=False).tolist()

            # Query Pinecone
            results = index.query(
                vector=[query_embedding],
                top_k=5,
                include_metadata=True
            )
            st.subheader("Top Search Results")
            for match in results['matches']:
                score = match['score']
                if score >= 0.3:  # Confidence threshold
                    details = match.get('metadata', {})
                    diagnosis = details.get('diagnosis', 'Unknown')
                    snippet = details.get('text_snippet', 'No snippet available.')

                    with st.container():
                        st.markdown(f"#### Classification: `{diagnosis}`")
                        st.caption(f"Semantic Match Score: {score:.3f}")
                        st.info(f"{snippet}")
                        st.divider()
    else:
        st.warning("Please enter a query to search.")