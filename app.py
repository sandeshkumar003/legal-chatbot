import streamlit as st
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ----------------------------------------------------
# Ensure folders exist
# ----------------------------------------------------
os.makedirs("data", exist_ok=True)


st.title("Pak Labor Law Assistant")
st.write("You can ask a question directly or upload documents → Build DB → Ask questions.")


# ----------------------------------------------------
# Upload files
# ----------------------------------------------------
files = st.file_uploader(
    "Upload PDFs or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if files:
    for f in files:
        path = os.path.join("data", f.name)
        with open(path, "wb") as out:
            out.write(f.read())
    st.success("Files uploaded.")


# ----------------------------------------------------
# Load files
# ----------------------------------------------------
def load_docs():
    docs = []
    for f in os.listdir("data"):
        path = os.path.join("data", f)
        if f.endswith(".txt"):
            docs.extend(TextLoader(path).load())
        elif f.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
    return docs


# ----------------------------------------------------
# Build Vector DB
# ----------------------------------------------------
if st.button("Build Knowledge Base"):
    with st.spinner("Processing documents..."):

        docs = load_docs()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        db = Chroma.from_documents(
            chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )

    st.success("Knowledge Base Ready!")


# ----------------------------------------------------
# Create retriever + LLM
# ----------------------------------------------------
def get_retriever():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    return db.as_retriever()


@st.cache_resource
def get_llm():
    return LlamaCpp(
        model_path="model.gguf",
        n_ctx=4096,
        max_tokens=256,
        temperature=0.1,
        n_threads=4,
    )


# ----------------------------------------------------
# Ask question
# ----------------------------------------------------
query = st.text_input("Ask your question:")

if query:
    retriever = get_retriever()
    llm = get_llm()

    prompt = PromptTemplate.from_template("""
    You are a helpful assistant. Use ONLY the context below to answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    with st.spinner("Thinking..."):
        answer = rag_chain.invoke(query)

    st.subheader("📘 Answer")
    st.write(answer)
