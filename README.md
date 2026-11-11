# AI-Powered PDF Chatbot

An interactive document assistant built with Python, Streamlit, LangChain, FAISS, and the Google Gemini API. This application allows users to upload PDF documents, ask context-based questions using Retrieval-Augmented Generation (RAG) with page references, generate concise summaries, and create practice quizzes.

## Features
- **PDF Upload & Text Extraction:** Supports single or multiple text-based PDF files and extracts text page-by-page using PyPDF.
- **Text Chunking & Vector Indexing:** Splits extracted text into smaller chunks and creates vector embeddings using Sentence Transformers (all-MiniLM-L6-v2), stored locally in a FAISS vector database.
- **Document Q&A (RAG):** Finds the most relevant document sections for a user's question and generates answers strictly from the PDF content along with source page numbers.
- **Document Summarization:** Generates a clear and structured summary of the uploaded PDF document.
- **Quiz Generator:** Automatically creates 5 Multiple Choice Questions (MCQs) and 3 short-answer questions with an answer key from the document text.

## Tech Stack
- **Language:** Python
- **Frontend UI:** Streamlit
- **Framework:** LangChain
- **LLM:** Google Gemini API (gemini-3.8-flash)
- **Embeddings:** HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store:** FAISS (faiss-cpu)
- **PDF Reader:** PyPDF

## How to Run Locally
1. Clone the repository and open the project folder in terminal.
2. Create a virtual environment and install the required packages using: pip install -r requirements.txt
3. Enter your free Google Gemini API key in the sidebar (or store it as GOOGLE_API_KEY in a .env file).
4. Start the application by running: streamlit run app.py

## Project Structure
- app.py - Main application source code
- requirements.txt - List of required Python packages
- .gitignore - Specifies untracked files to ignore
- README.md - Project documentation and setup guide