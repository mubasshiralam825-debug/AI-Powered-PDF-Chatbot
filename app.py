import os
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.set_page_config(page_title="PDF Document Assistant", layout="wide")
st.title("AI-Powered PDF Chatbot")
st.write("Upload your PDF documents to ask questions, generate summaries, and create practice quizzes.")

# helper function to extract clean text from model response
def format_output(content):
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)

# session variables
if "db" not in st.session_state:
    st.session_state.db = None
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "history" not in st.session_state:
    st.session_state.history = []

# sidebar setup
st.sidebar.header("Settings & Upload")
user_api_key = st.sidebar.text_input("Gemini API Key", type="password")
api_key = user_api_key if user_api_key else os.getenv("GOOGLE_API_KEY")

pdf_files = st.sidebar.file_uploader("Select PDF files", type=["pdf"], accept_multiple_files=True)

if st.sidebar.button("Process Documents"):
    if not pdf_files:
        st.sidebar.warning("No PDF file selected.")
    else:
        with st.spinner("Reading PDF and building index..."):
            extracted_pages = []
            combined_text = ""
            
            for f in pdf_files:
                try:
                    pdf_reader = PdfReader(f)
                    for idx, page in enumerate(pdf_reader.pages):
                        content = page.extract_text()
                        if content and content.strip():
                            clean_content = " ".join(content.split())
                            extracted_pages.append({
                                "content": clean_content,
                                "meta": {"file": f.name, "page": idx + 1}
                            })
                            combined_text += f"\n[Page {idx + 1}] {clean_content}"
                except Exception as err:
                    st.error(f"Could not read {f.name}: {err}")
            
            if len(extracted_pages) == 0:
                st.error("Could not extract any text from the PDF. Please check if it is a valid text-based PDF.")
            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunk_list = []
                meta_list = []
                
                for item in extracted_pages:
                    splits = text_splitter.split_text(item["content"])
                    for s in splits:
                        chunk_list.append(s)
                        meta_list.append(item["meta"])
                
                embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.db = FAISS.from_texts(chunk_list, embed_model, metadatas=meta_list)
                st.session_state.raw_text = combined_text
                st.sidebar.success("Processing complete!")

if st.sidebar.button("Reset"):
    st.session_state.db = None
    st.session_state.raw_text = ""
    st.session_state.history = []
    st.rerun()

# main layout tabs
qa_tab, sum_tab, quiz_tab = st.tabs(["Document Q&A", "Summarization", "Quiz Generator"])

with qa_tab:
    st.subheader("Ask a Question")
    query = st.text_input("Type your question here:")
    
    if st.button("Submit Question"):
        if not api_key:
            st.error("API key is required.")
        elif st.session_state.db is None:
            st.warning("Please upload and process a PDF first.")
        elif query:
            with st.spinner("Fetching answer..."):
                matched_docs = st.session_state.db.similarity_search(query, k=4)
                context_data = ""
                ref_list = []
                
                for doc in matched_docs:
                    ref = f"{doc.metadata['file']} - Page {doc.metadata['page']}"
                    if ref not in ref_list:
                        ref_list.append(ref)
                    context_data += f"Source ({ref}):\n{doc.page_content}\n\n"
                
                qa_template = """Answer the question based only on the following context. 
If the context does not contain the answer, reply with "Information not found in the document."

Context:
{context}

Question: {question}
Answer:"""
                
                prompt = PromptTemplate(template=qa_template, input_variables=["context", "question"])
                model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=api_key, temperature=0.2)
                
                chain = prompt | model
                result = chain.invoke({"context": context_data, "question": query})
                
                st.session_state.history.append({
                    "q": query,
                    "a": format_output(result.content),
                    "refs": ref_list
                })

    for item in reversed(st.session_state.history):
        st.markdown(f"**Q:** {item['q']}")
        st.markdown(f"**A:** {item['a']}")
        st.caption(f"References: {', '.join(item['refs'])}")
        st.markdown("---")

with sum_tab:
    st.subheader("Document Summary")
    if st.button("Create Summary"):
        if not api_key:
            st.error("API key is required.")
        elif not st.session_state.raw_text:
            st.warning("No processed PDF found.")
        else:
            with st.spinner("Summarizing document..."):
                model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=api_key, temperature=0.3)
                prompt_text = f"Write a clear and concise summary of the following document content:\n\n{st.session_state.raw_text[:20000]}"
                output = model.invoke(prompt_text)
                st.markdown(format_output(output.content))

with quiz_tab:
    st.subheader("Practice Questions")
    if st.button("Generate Questions"):
        if not api_key:
            st.error("API key is required.")
        elif not st.session_state.raw_text:
            st.warning("No processed PDF found.")
        else:
            with st.spinner("Creating quiz..."):
                model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=api_key, temperature=0.4)
                prompt_text = (
                    "Create 5 Multiple Choice Questions (MCQs) and 3 short-answer questions "
                    "from the text below. Include the correct answers at the bottom.\n\n"
                    f"{st.session_state.raw_text[:20000]}"
                )
                output = model.invoke(prompt_text)
                st.markdown(format_output(output.content))