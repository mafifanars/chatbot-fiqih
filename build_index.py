# build_index.py
# Memproses semua PDF di dalam sebuah folder dan menyimpannya 
# ke dalam satu indeks FAISS dengan nama kustom.

import os
import argparse
from glob import glob
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

def parse_args() -> argparse.Namespace:
    """Mendefinisikan dan mem-parsing argumen command-line."""
    parser = argparse.ArgumentParser(description="Build a FAISS index from all PDFs in a directory.")
    parser.add_argument("--pdf_dir", default="data_pdfs", help="Folder that contains the PDFs to index.")
    parser.add_argument("--out_dir", default="vectorstore", help="Folder to save the FAISS index files.")
    parser.add_argument("--index_name", default="fiqih_faiss", help="Base name for the output index files (.faiss, .pkl).")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for splitting text.")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Chunk overlap for splitting.")
    parser.add_argument("--embed_model", default="models/text-embedding-004", help="Gemini embedding model id.")
    return parser.parse_args()

def main():
    """Fungsi utama untuk menjalankan proses indexing."""
    args = parse_args()

    # Mengambil API key dari environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)
        
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Error configuring Google AI: {e}")
        sys.exit(1)

    # Mencari semua file PDF di direktori yang ditentukan
    print(f"Searching for PDF files in '{args.pdf_dir}'...")
    pdf_files = glob(os.path.join(args.pdf_dir, "*.pdf"))

    if not pdf_files:
        print(f"Error: No PDF files found in '{args.pdf_dir}'. Please check the directory.")
        sys.exit(1)
        
    print(f"Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        print(f"  - {os.path.basename(pdf)}")

    # 1. Muat halaman dari semua PDF
    all_docs = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"Warning: Could not load {pdf_path}. Error: {e}")
    
    if not all_docs:
        print("Error: Failed to load any documents from the PDF files.")
        sys.exit(1)

    print(f"\nLoaded a total of {len(all_docs)} pages from all PDF files.")

    # 2. Pecah teks menjadi chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} text chunks.")

    # 3. Inisialisasi model embedding
    print(f"Initializing embedding model: {args.embed_model}")
    embeddings = GoogleGenerativeAIEmbeddings(model=args.embed_model)

    # 4. Bangun indeks FAISS dan simpan
    print("Building FAISS index... (this may take a while)")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Memastikan direktori output ada
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Menyimpan indeks dengan nama kustom ke direktori output
    vectorstore.save_local(folder_path=args.out_dir, index_name=args.index_name)
    
    print(f"\nFAISS index successfully built and saved in '{args.out_dir}' with base name '{args.index_name}'")

if __name__ == "__main__":
    main()