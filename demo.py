import os
import requests
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer

# Constants
PDF_PATH = "./resume.pdf"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "rag_collection"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("Please set the TOGETHER_API_KEY environment variable")


def pdf_reader(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None


def textsplitter(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    return [doc.page_content for doc in splitter.create_documents([text])]


def init_vector_db(db_path=DB_PATH, collection_name=COLLECTION_NAME):
    client = chromadb.PersistentClient(path=db_path)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    class SBERTEmbeddingFunction:
        def __call__(self, input):
            if isinstance(input, str):
                input = [input]
            return model.encode(input).tolist()

        def name(self):
            return "sbert-mini"

    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=SBERTEmbeddingFunction()
    )


def add_documents_to_collection(collection, chunks):
    if collection.count() == 0:
        ids = [f"pdf_chunk_{i}" for i in range(len(chunks))]
        print(f"Adding {len(chunks)} chunks to the vector DB...")
        collection.add(documents=chunks, ids=ids)
    else:
        print("Collection already populated.")


def query_collection(collection, query, n_results=2):
    return collection.query(query_texts=[query], n_results=n_results)["documents"][0]


def generate_answer_with_together(prompt):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.2,
        "stop": ["\n\n"]
    }
    response = requests.post("https://api.together.xyz/v1/completions", json=data, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["text"].strip()
    print("Error:", response.status_code, response.text)
    return None


def simple_planner(query):
    q = query.lower()
    if "summarize" in q:
        return "summarize"
    elif "hr" in q and "action" in q:
        return "extract_hr"
    return "general"


def tool_summarize(text):
    return generate_answer_with_together(f"Summarize the following text:\n\n{text}\n\nSummary:")


def tool_extract_hr_tasks(text):
    return generate_answer_with_together(
        f"From the following text, extract action items relevant to the HR department:\n\n{text}\n\nItems:"
    )


def tool_answer_question(text, question):
    prompt = f"""
Using the following context, answer the question. If the answer is not in the context,
say you don't know.

Context:
{text}

Question: {question}

Answer:
"""
    return generate_answer_with_together(prompt)


def main():
    print("Reading PDF...")
    text = pdf_reader(PDF_PATH)
    if not text:
        print("No text extracted. Exiting.")
        return

    print("Splitting text into chunks...")
    chunks = textsplitter(text)
    print("Initializing vector database...")
    collection = init_vector_db()
    print("Adding chunks to vector database...")
    add_documents_to_collection(collection, chunks)

    query = input("\nEnter your question: ")
    print(f"\nSearching for relevant chunks for query: '{query}'")
    retrieved_chunks = query_collection(collection, query, n_results=2)

    context = "\n".join(retrieved_chunks)
    action = simple_planner(query)
    print(f"\n[Agent Plan]: Detected intent → {action}")

    if action == "summarize":
        answer = tool_summarize(context)
    elif action == "extract_hr":
        answer = tool_extract_hr_tasks(context)
    else:
        answer = tool_answer_question(context, query)

    print("\n--- Agentic Answer ---")
    print(answer if answer else "Failed to generate answer.")


if __name__ == "__main__":
    main()
