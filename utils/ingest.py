import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import os

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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')
    pdf_path = os.path.join(data_dir, "HR-Policy.pdf")
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    out_path = os.path.join(data_dir, "hr_chunks.json")
    with open(out_path, "w") as f:
        json.dump(chunks, f)
    print("Chunks saved to data/hr_chunks.json")
