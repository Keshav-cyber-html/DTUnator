# DTUnator

DTUnator is an on-demand AI-powered doubt-solving assistant designed specifically for students of Delhi Technological University (DTU). It helps students resolve academic doubts by retrieving answers directly from verified lecture notes and course material, ensuring accuracy and relevance.

DTUnator follows a Retrieval-Augmented Generation (RAG) architecture. It does not rely on the language model’s general knowledge. Instead, every response is generated strictly from documents retrieved at query time, making the system suitable for academic use where correctness and source grounding are essential.

## How DTUnator Works

DTUnator operates fully on-demand. Each user query triggers a fresh retrieval and reasoning pipeline:

1. The user submits a question  
2. The query is rewritten to remove ambiguity and improve clarity  
3. A hypothetical supporting document (HyDE) is generated to enhance semantic retrieval  
4. Relevant chunks are retrieved from a Weaviate vector database using semantic search  
5. Retrieved documents are re-ranked using a cross-encoder for higher relevance  
6. The final answer is generated using only the retrieved documents  

A verification step ensures the response is fully supported by the retrieved context.  
If sufficient information is not available, DTUnator explicitly states that it cannot answer the question rather than guessing.

## Key Characteristics

- Fully on-demand query processing  
- Strict document-grounded responses  
- No hallucinations or external knowledge usage  
- Transparent handling of insufficient data  
- Designed for academic reliability  

## Technology Stack

- **Language Model:** Google Gemini 2.5 Flash  
- **Vector Database:** Weaviate (Cloud)  
- **Embeddings:** all-MiniLM-L6-v2  
- **Re-ranking:** MS MARCO MiniLM Cross-Encoder  
- **Backend:** Python  
- **Libraries:** LangChain, SentenceTransformers  
- **Interface:** Command-line (Streamlit integration planned)  

## Purpose

DTUnator is built to assist DTU students with exam preparation, concept clarification, and fast access to lecture content through an intelligent, reliable, and on-demand AI system grounded entirely in institutional material.
