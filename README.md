
---

#  **NoLimit DS Technical Test — RAG Chatbot (Reinforcement Learning Papers)**

**Author:** Muh Rivandy Setiawan
**Role:** Data Scientist Applicant
**Project:** Retrieval-Augmented Generation (RAG) Chatbot for Reinforcement Learning Research
**Framework:** Python, Streamlit, FAISS, Sentence-Transformers

---

##  **Project Overview**

This repository contains my solution for the **NoLimit Data Science Technical Test**, where I built a **retrieval-based Q&A system with citations** using a lightweight RAG pipeline.

The system processes three RL-related research papers:

1. **DAPO — An Open-Source LLM Reinforcement Learning System at Scale**
2. **RLAC — Reinforcement Learning with Adversarial Critic for Free-Form Generation Tasks**
3. **RLVE — Scaling Up Reinforcement Learning for Language Models**

The RAG system is able to:

✔ Extract text from PDFs
✔ Perform **section-aware chunking**
✔ Generate **dense embeddings** using SentenceTransformers
✔ Build a **FAISS vector index**
✔ Retrieve the most relevant chunks
✔ Return **answers with clear citations** (file name + section)

This solution fulfils all the **Mandatory Requirements** from the assignment.

---

##  **System Architecture**

The pipeline consists of:

1. **PDFLoader** — load & clean document text
2. **SectionAwareChunker** — split into meaningful sentence blocks grouped by section
3. **EmbeddingGenerator** — generate vector embeddings
4. **VectorIndex (FAISS)** — index vectors for fast similarity search
5. **Retriever** — return relevant chunks from the index
6. **AnswerFormatter** — format results with citations
7. **Streamlit App (optional)** — for user-friendly interaction
8. **RAGPipeline** — end-to-end orchestrator for the whole process

---

##  **Flowchart (End-to-End Pipeline)**


<img width="761" height="201" alt="flowchart" src="https://github.com/user-attachments/assets/60e5c964-9a8e-4e35-8cd7-a020483aa540" />


Full version available in `flowchart.pdf`.

---

##  **Repository Structure**

```
nolimit-ds-test-rivandy/
│── data_sample/
│     ├── DAPO.pdf
│     ├── RLAC.pdf
│     └── RLVE.pdf
│── notebook/
│     └── testing_pipeline.ipynb
│── src/
│     ├── AnswerFormatter.py
│     ├── EmbeddingGenerator.py
│     ├── PDFLoader.py
│     ├── RAGPipeline.py
│     ├── Retriever.py
│     ├── StreamlitApp.py   # Streamlit UI (optional)
│     ├── TextChunker.py
│     ├── Utilities.py
│     └── VectorIndexFAISS.py
│── LICENSE
│── README.md
│── flowchart.pdf
│── flowchart.png
│── requirements.txt

```

---

##  **Dataset Source & License**

All PDF documents in this repo come from **arXiv.org**:

* [https://arxiv.org/abs/](https://arxiv.org/abs/)
* License: **CC BY 4.0** (Creative Commons Attribution)
* Documents are open-access and permitted for research & training usage.

Each PDF citation is also listed in metadata for transparency.

---

##  **Mandatory Requirements Checklist**

| Requirement                                                   | Status                                |
| ------------------------------------------------------------- | ------------------------------------- |
| Use HuggingFace Models (Transformers / Sentence-Transformers) | ✔                                     |
| Use Embeddings (FAISS / ANN)                                  | ✔                                     |
| Provide README + dataset source                               | ✔                                     |
| Section-aware Chunking                                        | ✔                                     |
| Flowchart PDF/PNG                                             | ✔                                     |
| Public GitHub Repository                                      | ✔                                     |
| Small sample PDF dataset                                      | ✔                                     |
| Streamlit/Flask Deployment (Bonus)                            | Optional — Streamlit version included |

---

##  **How to Run Locally**

### **1. Clone Repo**

```bash
git clone https://github.com/<your-username>/nolimit-ds-test-rivandy
cd nolimit-ds-test-rivandy
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run Streamlit App (recommended)**

```bash
streamlit run src/app.py
```

### **4. Or test pipeline without UI**

```python
from src.rag_pipeline import RAGPipeline
# load, chunk, embed, index, retrieve
```

---

##  **Deployment (Optional Bonus)**

This project includes a Streamlit UI (`src/app.py`) that can be deployed on:

### ✔ Hugging Face Spaces

### ✔ Streamlit Cloud

### ✔ Local deployment

Deployment instructions are provided inside the `app.py` file.

---

##  **Bonus Points Included**

The following bonus enhancements are implemented:

* ✔ Section-aware chunking (far better than naive chunking)
* ✔ Metadata-rich retrieval (section, document, sentence range)
* ✔ Clean citation formatting
* ✔ Modular pipeline design (`RAGPipeline`)
* ✔ Ready-to-deploy Streamlit UI
* ✔ Clear flowchart (PNG + PDF)
* ✔ Clean folder structure following ML engineering standards

---

##  **Author**

**Muh Rivandy Setiawan**
AI/ML & Data Science Enthusiast
2025

---

##  **Contact**

If you need clarification or further technical discussion, I am available to discuss it.

