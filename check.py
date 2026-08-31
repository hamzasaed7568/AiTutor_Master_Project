import fitz  
import os


BOOK_PATH = "dataset/raw_dataset/books/Introduction_To_Computer_Science.pdf"

print("PDF Book Load ho rahi hai...")

try:
    doc = fitz.open(BOOK_PATH)
    total_pages = len(doc)
    print(f"Success! Book mein total {total_pages} pages hain.\n")
    
    
    sample_text = ""
    for i in range(min(5, total_pages)):
        page = doc.load_page(i)
        sample_text += page.get_text()
        
    print("--- Book Ke Start Ka Text (Sample) ---")
    print(sample_text[:500])
    print("\n--------------------------------------")
    print("Congrats python have read the book successfully...")
    
except Exception as e:
    print(f"Error aagaya: {e}")
    print("Bhai path check kar lein, shayad file ka naam ya folder name thora different ho.")