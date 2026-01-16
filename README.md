# DTUnator

DTU students often struggle to find precise academic answers because information is scattered across too many documents, and generic AI chatbots aren't reliable for our specific curriculum.

DTUnator solves this. It’s an AI assistant built exclusively on verified DTU lecture materials. Instead of guessing based on general knowledge, it finds answers directly within official course documents. If the information isn’t there, it honestly says so. It also gives students/users the option to add their own files such as PDFs, youtube links, audio files, website links, etc. and ask their queries based on the source provided by them

The result is faster, trustworthy doubt resolution that helps students learn efficiently using material they can depend on.

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
