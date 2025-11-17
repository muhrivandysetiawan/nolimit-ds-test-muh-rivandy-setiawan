@st.cache_resource(show_spinner=True)
def load_pipeline():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    loader = PDFLoader()
    chunker = SectionAwareChunker()
    embedder = EmbeddingGenerator()
    vindex = VectorIndexFAISS()
    retriever = Retriever(
        embedding_model=embedder.model,
        vector_index=vindex,
        top_k=5,
        rewrite_query=False
    )
    formatter = AnswerFormatter()

    pipeline = RAGPipeline(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        vector_index=vindex,
        retriever=retriever,
        formatter=formatter,
        verbose=False
    )

    # --- load PDFs (local) ---
    loader.file_names = [
        os.path.join(DATA_DIR, "DAPO.pdf"),
        os.path.join(DATA_DIR, "RLAC.pdf"),
        os.path.join(DATA_DIR, "RLVE.pdf")
    ]

    loader.load_documents()

    # <<< IMPORTANT FIX
    pipeline.docs_loaded = True

    pipeline.make_chunks()
    pipeline.embed()
    pipeline.build_faiss()

    return pipeline

st.title("📘 RAG Chatbot for RL in LLM Research")
pipeline = load_pipeline()

query = st.text_input("Enter your question:")

if st.button("Search"):
    results = pipeline.ask(query, top_k=5, pretty=True)
    st.markdown(results)
