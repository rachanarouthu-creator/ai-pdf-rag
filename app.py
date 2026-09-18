import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


# ---------------------------------------
# 1. Page title
# ---------------------------------------

st.title("📚 AI PDF Question Answering System")

st.write("Ask questions from your PDF using RAG.")


# ---------------------------------------
# 2. Load PDF
# ---------------------------------------

loader = PyPDFLoader("documents/CN_UNIT-1_S_MAT.pdf")

documents = loader.load()


# ---------------------------------------
# 3. Split PDF into chunks
# ---------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)


# ---------------------------------------
# 4. Create embeddings
# ---------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# ---------------------------------------
# 5. Store chunks in ChromaDB
# ---------------------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)


# ---------------------------------------
# 6. Create Llama 3.2 model
# ---------------------------------------

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# ---------------------------------------
# 7. Question input
# ---------------------------------------

question = st.text_input("Enter your question:")


# ---------------------------------------
# 8. RAG process
# ---------------------------------------

if question:

    # Search for relevant information
    results = vectorstore.similarity_search(
        question,
        k=3
    )

    # Combine retrieved chunks
    context = "\n\n".join(
        result.page_content
        for result in results
    )

    # Create prompt
    prompt = f"""
    Answer the question using only the information
    provided in the context below.

    Context:
    {context}

    Question:
    {question}

    If the answer is not available in the context,
    say:
    "I could not find the answer in the PDF."
    """

    # Ask Llama
    response = llm.invoke(prompt)

    # Display answer
    st.subheader("Answer")

    st.write(response.content)