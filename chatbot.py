pip install transformers datasets faiss-cpu huggingface_hub


from huggingface_hub import login

login(token="hugging_face_token")


import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


pip install PyPDF2 faiss-cpu transformers datasets


import os
import faiss
import numpy as np
import PyPDF2
from transformers import AutoTokenizer, AutoModel
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def read_pdf(file_path="/content/DNA.pdf"):
    with open("/content/DNA.pdf", "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text.split("\n")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
print("Model and tokenizer loaded successfully")


def embed_texts(texts, max_length=512):
    print(f"Embedding {len(texts)} texts...")
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    print(f"Generated embeddings of shape: {embeddings.shape}")
    return embeddings

def chunk_text(text, chunk_size=100):
    print(f"Chunking text into pieces of size {chunk_size}...")
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"Created {len(chunks)} chunks")
    return chunks

#file path of pdf should be modified according to usecase
pdf_file_path = "/content/DNA.pdf"


pdf_text = read_pdf(pdf_file_path)
print(f"Extracted {len(pdf_text)} lines of text from PDF")


pdf_chunks = chunk_text(pdf_text, chunk_size=50)

pdf_embeddings = []
for chunk in pdf_chunks:
    embeddings = embed_texts(chunk)
    pdf_embeddings.append(embeddings)


pdf_embeddings = np.concatenate(pdf_embeddings, axis=0)
print(f"Final embeddings shape: {pdf_embeddings.shape}")


index = faiss.IndexFlatL2(pdf_embeddings.shape[1])
index.add(pdf_embeddings)
print("FAISS index created and embeddings added")


def retrieve_relevant_docs(query, k=3):
    query_embedding = embed_texts([query])
    print(f"Searching FAISS index with query: '{query}'...")
    distances, indices = index.search(query_embedding, k)
    print(f"Found indices: {indices[0]} with distances: {distances[0]}")
    return [pdf_text[i] for i in indices[0]]
query = " What is DNA made of ?"
relevant_docs = retrieve_relevant_docs(query)


print("Relevant documents:")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}: {doc}")


from transformers import pipeline


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_docs(docs, max_length=50):
    text = " ".join(docs)
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]['summary_text']


summary = summarize_docs(relevant_docs)
print("Summary:", summary)
