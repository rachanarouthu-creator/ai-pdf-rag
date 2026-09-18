
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from huggingface_hub import InferenceClient


# ---------------------------------------
# 1. Page title
# ---------------------------------------

st.title("📚 AI PDF Question Answering System")

st.write("Ask questions from your PDF using RAG.")


# ---------------------------------------
# 2. Hugging Face connection
# ---------------------------------------

hf_token = st.secrets["HF_TOKEN"]

hf_client = InferenceClient(
    token=hf_token
)


# ---------------------------------------
# 3. Hugging Face Embeddings
# ---------------------------------------

class HuggingFaceEmbeddings(Embeddings):

    def embed_documents(self, texts):

        embeddings = hf_client.feature_extraction(
            texts,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )

        return embeddings.tolist()

    def embed_query(self, text):

        embedding = hf_client.feature_extraction(
            text,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )

        return embedding.tolist()


# ---------------------------------------
# 4. Load PDF
# ---------------------------------------

loader = PyPDFLoader(
    "documents/CN_UNIT-1_S_MAT.pdf"
)

documents = loader.load()

st.write(f"PDF loaded: {len(documents)} pages")


# ---------------------------------------
# 5. Split PDF into chunks
# ---------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

st.write(f"Created {len(chunks)} text chunks")


# ---------------------------------------
# 6. Create embeddings
# ---------------------------------------

embeddings = HuggingFaceEmbeddings()


# ---------------------------------------
# 7. Store chunks in ChromaDB
# ---------------------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)


# ---------------------------------------
# 8. Question input
# ---------------------------------------

question = st.text_input(
    "Enter your question:"
)


# ---------------------------------------
# 9. RAG process
# ---------------------------------------

if question:

    # Retrieve relevant chunks
    results = vectorstore.similarity_search(
        question,
        k=3
    )

    # Combine retrieved information
    context = "\n\n".join(
        result.page_content
        for result in results
    )


    # -----------------------------------
    # 10. Create prompt
    # -----------------------------------

    prompt = f"""
Answer the question using ONLY the information
provided in the context below.

Context:
{context}

Question:
{question}

If the answer is not available in the context,
say:

"I could not find the answer in the PDF."
"""


    # -----------------------------------
    # 11. Ask Hugging Face model
    # -----------------------------------

    response = hf_client.chat_completion(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=300,
        temperature=0
    )


    # -----------------------------------
    # 12. Display answer
    # -----------------------------------

    st.subheader("Answer")

    st.write(
        response.choices[0].message.content
    )
