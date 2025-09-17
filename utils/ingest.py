import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks

if __name__ == "__main__":
    pdf_path = "data/HR-Policy.pdf"
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    with open("hr_chunks.json", "w") as f:
        json.dump(chunks, f)
    print("Chunks saved to hr_chunks.json")
