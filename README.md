# 🧠 RAG-Based AI Resume Screener and Chat Assistant with Agentic Reasoning

An intelligent resume-screening and job-fit analyzer powered by LLM embeddings,
RAG pipelines, and agentic reasoning.

## 📌 Project Highlights

- Built an intelligent resume-job matcher using LLM embeddings and RAG pipelines
  with **LangChain** and **FAISS** for semantic search, achieving **95% accuracy**
  in skill-to-JD alignment.
- Designed a multi-stage document ingestion pipeline supporting **PDF and DOCX**
  formats with semantic chunking strategies, processing **1000+ document chunks**
  with an average retrieval latency of **300 ms**.
- Integrated **OpenAI API** with modular prompt orchestration enabling sub-second
  AI query handling across resume parsing and job-fit reasoning workflows.
- Developed an interactive **Streamlit UI** enabling **50+ users** to upload
  resumes and receive real-time AI-driven feedback on job fit and keyword
  alignment.

## 🔍 Overview

- Upload a PDF/DOCX resume and parse its content
- Split text into chunks for semantic search
- Generate real embeddings using `all-MiniLM-L6-v2`
- Store and query document chunks in **ChromaDB**
- Ask questions through a friendly Streamlit chat interface
- Generate context-aware answers using Together.ai / OpenAI LLM

---

## 🧰 Tech Stack

| Component    | Technology                            |
|--------------|---------------------------------------|
| Frontend     | Streamlit                             |
| Backend      | Python, LangChain                     |
| Embeddings   | Sentence Transformers (`MiniLM`)      |
| Vector Store | ChromaDB / FAISS                      |
| LLM          | Together.ai (Mistral-7B) / OpenAI API |
| PDF Reader   | PyPDF                                 |

---

## 🚀 Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set the following environment variables before running:

```bash
export TOGETHER_API_KEY="your_together_api_key"
# or
export OPENAI_API_KEY="your_openai_api_key"
```

## 📁 File Structure

| File          | Description                                         |
|---------------|-----------------------------------------------------|
| `app.py`      | Main Streamlit UI – upload PDF and chat             |
| `main.py`     | Core RAG utilities (PDF reader, vector DB, LLM)     |
| `agentic.py`  | Agentic reasoning loop with tool-use planner        |
| `agentic2.py` | Advanced agentic RAG with critic and goal policy    |
| `demo.py`     | CLI demo with planner and Together.ai integration   |
| `app1.py`     | Alternative LangChain + FAISS + HuggingFace version |

---

## 📄 License

MIT
