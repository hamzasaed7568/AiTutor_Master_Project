import fitz  # PyMuPDF library ka import name fitz hota hai
import os

# PDF Book ka path
BOOK_PATH = "dataset/raw_dataset/books/Introduction_To_Computer_Science.pdf"

print("PDF Book Load ho rahi hai...")

try:
    # PDF ko open karna
    doc = fitz.open(BOOK_PATH)
    total_pages = len(doc)
    print(f"Success! Book mein total {total_pages} pages hain.\n")
    
    # Testing ke liye sirf pehle 5 pages ka text nikalte hain
    sample_text = ""
    for i in range(min(5, total_pages)):
        page = doc.load_page(i)
        sample_text += page.get_text()
        
    print("--- Book Ke Start Ka Text (Sample) ---")
    # Sirf pehle 500 characters print karte hain taake terminal na bhar jaye
    print(sample_text[:500])
    print("\n--------------------------------------")
    print("Congrats python have read the book successfully...")
    
except Exception as e:
    print(f"Error aagaya: {e}")
    print("Bhai path check kar lein, shayad file ka naam ya folder name thora different ho.")