
# from langchain_community.document_loaders import PyPDFLoader

# # Location of our sample PDF
# pdf_path = "data/sample.pdf"

# # Load the PDF
# loader = PyPDFLoader(pdf_path)
# pages = loader.load()

# print("PDF loaded successfully!")
# print("Total pages:", len(pages))

# # Display the first page
# if pages:
#     print("\nFirst page text:\n")
#     print(pages[0].page_content[:2000])


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

PDF_PATH = r"C:\Users\bachh\OneDrive\Desktop\SmartDoc-AI\data\sample.pdf"
DB_PATH = "faiss_index"

print("Loading PDF...")

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"PDF loaded: {len(documents)} pages")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} text chunks")

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating FAISS vector database...")

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(DB_PATH)

print(f"Done! Vector database saved to {DB_PATH}")
