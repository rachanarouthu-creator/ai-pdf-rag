from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# 1. Load PDF
loader = PyPDFLoader("documents/CN_UNIT-1_S_MAT.pdf")
documents = loader.load()

print("Number of pages:", len(documents))


# 2. Split PDF into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# 3. Create embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# 4. Create/load ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)


# 5. Ask a question
question = "What is computer networking?"


# 6. Search for relevant chunks
results = vectorstore.similarity_search(
    question,
    k=3
)


# 7. Display the results
print("\nQuestion:")
print(question)

print("\nRelevant information from PDF:")

for i, result in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print(result.page_content)
# 8. Create the local LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)
# 9. Combine the retrieved chunks
context = "\n\n".join(
    result.page_content for result in results
)
# 10. Create the prompt
prompt = f"""
Answer the question using only the information provided in the context below.

Context:
{context}

Question:
{question}

If the answer is not available in the context, say:
"I could not find the answer in the PDF."
"""
# 11. Ask Llama to generate the answer
response = llm.invoke(prompt)
# 12. Display the final answer
print("\nFinal Answer:")
print(response.content)