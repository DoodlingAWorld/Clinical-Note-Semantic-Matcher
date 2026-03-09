import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="BioMedical AI Search", page_icon="🧬", layout="centered")

st.title("🧬 Medical Literature Semantic Search")
st.markdown("""
Clinical Discovery Engine — hybrid PubMedBERT + BM25 retrieval with GPT-4o-mini synthesis.
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
    """Scale dense by alpha and sparse by (1-alpha) before querying."""
    hsparse = {
        "indices": sparse["indices"],
        "values": [v * (1 - alpha) for v in sparse["values"]],
    }
    return [v * alpha for v in dense], hsparse


def get_rag_answer(query: str, matches: list) -> str:
    """
    Feed the retrieved documents to GPT-4o-mini and return a synthesized answer.

    The prompt constrains the LLM to ONLY use the provided documents.
    This is the core RAG guarantee: no hallucinations from outside the retrieved context.
    Citation format [Doc N] maps directly to the numbered source cards in the UI.
    """
    # Format retrieved snippets as a numbered list for the LLM
    docs_text = ""
    for i, match in enumerate(matches, 1):
        details = match.get("metadata", {})
        diagnosis = details.get("diagnosis", "Unknown")
        snippet = details.get("text_snippet", "")
        docs_text += f"[Doc {i}] ({diagnosis})\n{snippet}\n\n"

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical AI research assistant. "
                    "Answer the user's query using ONLY the information in the provided documents. "
                    "Cite sources using [Doc N] after each claim. "
                    "If the documents do not contain enough information to answer, say so explicitly. "
                    "Be concise — 3 to 5 sentences."
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nDocuments:\n{docs_text}",
            },
        ],
        temperature=0.2,  # Low temp: factual synthesis, not creative writing
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


with st.spinner("Loading AI models and vector database..."):
    model = load_model()
    bm25 = load_bm25()
    index = init_pinecone()

st.write("---")

query = st.text_input(
    "Enter clinical symptoms or a research topic:",
    placeholder="e.g., patient presenting with pulmonary nodules and severe coughing..."
)

alpha = st.slider(
    "Search mode",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05,
    help="Semantic: understands meaning and synonyms. Keyword: exact term matching. "
         "Hybrid blends both — useful for specific gene names, drug names, or technical identifiers."
)
st.caption(f"← Pure keyword (BM25)  |  alpha = {alpha:.2f}  |  Pure semantic (PubMedBERT) →")

if st.button("Search Literature", type="primary"):
    if query:
        # Step 1: Hybrid retrieval
        with st.spinner("Searching vector database..."):
            hdense, hsparse = hybrid_scale(
                model.encode(query, show_progress_bar=False).tolist(),
                bm25.encode_queries(query),
                alpha,
            )
            results = index.query(
                vector=hdense,
                sparse_vector=hsparse,
                top_k=5,
                include_metadata=True,
            )

        matches = results.get("matches", [])

        if not matches:
            st.warning("No results found. Try adjusting the search mode or rephrasing your query.")
        else:
            # Step 2: RAG synthesis
            with st.spinner("Synthesizing answer with GPT-4o-mini..."):
                answer = get_rag_answer(query, matches)

            # Display synthesized answer
            st.subheader("AI Answer")
            st.success(answer)

            # Display source cards so the user can verify every claim
            st.subheader("Sources")
            for i, match in enumerate(matches, 1):
                details = match.get("metadata", {})
                diagnosis = details.get("diagnosis", "Unknown")
                snippet = details.get("text_snippet", "No snippet available.")
                score = match["score"]

                with st.container():
                    st.markdown(f"**[Doc {i}]** `{diagnosis}`")
                    st.caption(f"Hybrid score: {score:.4f}  |  alpha = {alpha:.2f}")
                    st.info(snippet)
                    st.divider()
    else:
        st.warning("Please enter a query to search.")
