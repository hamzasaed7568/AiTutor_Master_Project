import os
import fitz  
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

BOOK_PATH = "dataset/raw_dataset/books/Introduction_To_Computer_Science.pdf"
CHROMA_DIR = "./chroma_db"

def clean_text(text):
    """Text cleaning function (extra spaces aur newlines)"""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

print("Step 1: PDF Book is loading...")
doc = fitz.open(BOOK_PATH)
full_text = ""

for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    raw_text = page.get_text()
    full_text += clean_text(raw_text) + " "

print(f"Book read successfully! Total characters: {len(full_text)}")

print("\nStep 2: Text Chunking start ho rahi hai...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = text_splitter.split_text(full_text)
print(f"Total Chunks ban gaye: {len(chunks)}")

print("\nStep 3: Embedding Model are loading (Free HuggingFace Model)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("\nStep 4: Vector Database (ChromaDB) is making.")
print("⏳ Please wait, full is book is processing and it gonna take time according to your PC...")

vector_db = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)

print("\n🎉 Congrats! Book preprocessing and Vector DB setup completed 100%.")