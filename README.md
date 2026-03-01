# SemanticSearch
Built a weighted vector search engine using PubMedBERT and Pinecone. 
Standard semantic search struggles with dense medical documents by treating all words equally. 
To solve this, experimented with a weighted embedding pipeline that prioritizes the clinical diagnosis (5x weight) and abstract summary (3x weight) over the general text body, ensuring that highly relevant medical literature surfaces first.
