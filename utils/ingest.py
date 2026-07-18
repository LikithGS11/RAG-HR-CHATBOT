import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import os


def extract_pages_from_pdf(pdf_path):
    """Return a list of (page_number, text) tuples, 1-indexed."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((page_no, text))
    return pages


def chunk_pages(pages, chunk_size=1000, chunk_overlap=200):
    """Split each page's text into chunks, tagging every chunk with its page.

    Chunking per page keeps page attribution exact (at the cost of not
    spanning chunks across page boundaries)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for page_no, text in pages:
        for piece in splitter.split_text(text):
            chunks.append({"text": piece, "page": page_no})
    return chunks


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')
    pdf_path = os.path.join(data_dir, "HR-Policy.pdf")
    pages = extract_pages_from_pdf(pdf_path)
    chunks = chunk_pages(pages)
    out_path = os.path.join(data_dir, "hr_chunks.json")
    with open(out_path, "w") as f:
        json.dump(chunks, f)
    print(f"{len(chunks)} chunks (with page numbers) saved to data/hr_chunks.json")
