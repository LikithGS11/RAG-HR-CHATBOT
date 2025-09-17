# RAG HR Chatbot

## Project Overview
The RAG HR Chatbot is a Retrieval-Augmented Generation (RAG) chatbot designed to assist with HR policies and related queries. It leverages a combination of FAISS vector search, BM25 re-ranking, and Groq LLM to provide accurate and context-aware answers based on your HR policy documents.

The system consists of:
- A Flask backend API that handles query processing, retrieval, and interaction with the Groq LLM.
- A Streamlit frontend UI for an interactive chat experience.
- Preprocessing utilities to extract, chunk, embed, and index HR policy documents.
- Docker support for easy deployment.

## Features
- Hybrid retrieval using FAISS and BM25 for improved relevance.
- Session-based chat history with caching for performance.
- Easy ingestion of HR policy PDFs.
- Modern, responsive frontend UI with Streamlit.
- Secure API key management via environment variables.

## Prerequisites
- Python 3.8 or higher
- Docker (optional, for containerized deployment)
- Groq API key (sign up at Groq.ai)

## Local Development Setup

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd RAG-HR-Chatbot
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place your HR policy PDF in the `data/` directory as `HR-Policy.pdf`.

5. Ingest and preprocess the document:
   ```bash
   python utils/ingest.py
   python utils/embeddings.py
   python utils/faiss_index.py
   ```

6. Set your Groq API key in a `.env` file at the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

7. Run the backend API:
   ```bash
   python app/backend/app.py
   ```

8. In a separate terminal, run the frontend UI:
   ```bash
   streamlit run frontend/app.py
   ```

9. Open your browser at `http://localhost:8501` to interact with the chatbot.

## Docker Usage

### Build and Run Locally
```bash
docker build -t rag-hr-chatbot .
docker run -p 8501:8501 rag-hr-chatbot
```

### Using Docker Compose
```bash
docker-compose up --build
```
This will start both backend and frontend services with proper environment variable injection.

## Environment Variables
- `GROQ_API_KEY`: Your Groq API key for accessing the LLM service.

## Project Structure
```
.
├── app/
│   ├── backend/
│   │   └── app.py          # Flask backend API
│   └── frontend/
│       └── app.py          # Streamlit frontend UI
├── data/
│   └── HR-Policy.pdf       # HR policy document
├── models/                 # Serialized embeddings, indexes, and chunks
├── utils/
│   ├── embeddings.py       # Create embeddings from chunks
│   ├── faiss_index.py      # Build FAISS and BM25 indexes
│   └── ingest.py           # Extract and chunk text from PDF
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Security Notes
- Never commit your `.env` file or API keys to version control.
- Regularly update dependencies to patch vulnerabilities.
- Expose only necessary ports in production environments.
- Use secure network configurations when deploying.

## Contact
Developed by Likith at Datamites.
