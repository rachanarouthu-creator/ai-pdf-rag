from langchain_community.document_loaders import PyPDFLoader

print("Starting...")

loader = PyPDFLoader("documents/CN_UNIT-1_S_MAT.pdf")

print("Loader created")

documents = loader.load()

print("PDF loaded")
print("Number of pages:", len(documents))