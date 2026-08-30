import os
import fitz  # PyMuPDF
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Paths set karein
BOOK_PATH = "dataset/raw_dataset/books/Introduction_To_Computer_Science.pdf"
CHROMA_DIR = "./chroma_db"

def clean_text(text):
    """Text ko clean karne ka function (extra spaces aur newlines remove karna)"""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

print("Step 1: PDF Book load aur read ho rahi hai...")
doc = fitz.open(BOOK_PATH)
full_text = ""

# Poori book read aur clean kar rahe hain
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    raw_text = page.get_text()
    full_text += clean_text(raw_text) + " "

print(f"Book successfuly read ho gayi! Total characters: {len(full_text)}")

print("\nStep 2: Text Chunking start ho rahi hai...")
# Chunking strategy: 1000 characters ka chunk, 100 characters overlap (context bachane ke liye)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = text_splitter.split_text(full_text)
print(f"Total Chunks ban gaye: {len(chunks)}")

print("\nStep 3: Embedding Model load ho raha hai (Free HuggingFace Model)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("\nStep 4: Vector Database (ChromaDB) ban raha hai.")
print("⏳ Please wait, poori book process hone mein aapke PC ke hisaab se 2 se 5 minute lag sakte hain...")

# Chunks ko ChromaDB mein hamesha ke liye save kar rahe hain
vector_db = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)

print("\n🎉 Mubarak ho! Book ki preprocessing aur Vector DB setup 100% complete ho gaya hai.")