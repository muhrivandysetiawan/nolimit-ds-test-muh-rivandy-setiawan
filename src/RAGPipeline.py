class RAGPipeline:
    def __init__(self,
                 loader,
                 chunker,
                 embedder,
                 vector_index,
                 retriever,
                 formatter,
                 verbose=True):

        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.index = vector_index
        self.retriever = retriever
        self.formatter = formatter
        self.verbose = verbose
        self.docs_loaded = False
        self.chunks_ready = False
        self.embeddings_ready = False
        self.index_ready = False

    def load_pdfs(self):
        if self.verbose:
            print("\n[1] LOADING PDFs...")
        self.loader.upload_pdfs()
        self.loader.load_documents()
        self.docs_loaded = True
        if self.verbose:
            print("[✓] PDFs loaded.\n")
        return self.loader.get_doc_map()

    def make_chunks(self):
        if not self.docs_loaded:
            raise Exception("Run load_pdfs() first.")
        if self.verbose:
            print("\n[2] CHUNKING...")
        self.chunker.process_documents(self.loader.get_doc_map())
        self.chunks_ready = True
        if self.verbose:
            print("[✓] Chunks created.\n")
        return self.chunker.get_chunk_meta()

    def embed(self):
        if not self.chunks_ready:
            raise Exception("Run make_chunks() first.")
        if self.verbose:
            print("\n[3] ENCODING EMBEDDINGS...")
        chunks = self.chunker.get_chunks()
        meta = self.chunker.get_chunk_meta()
        self.embedder.encode_chunks(chunks, meta)
        self.embeddings_ready = True
        if self.verbose:
            print("[✓] Embeddings done.\n")
        return self.embedder.embeddings

    def build_faiss(self):
        if not self.embeddings_ready:
            raise Exception("Run embed() first.")
        if self.verbose:
            print("\n[4] BUILDING FAISS INDEX...")
        self.index.build(
            embeddings=self.embedder.embeddings,
            chunk_texts=self.embedder.chunk_texts,
            chunk_meta=self.embedder.chunk_meta
        )
        self.index_ready = True
        if self.verbose:
            print("[✓] FAISS index ready.\n")

    def ask(self, query, top_k=5, pretty=False):
        if not self.index_ready:
            raise Exception("Run build_faiss() first.")

        if self.verbose:
            print(f"\n[5] QUERY → {query}")

        retrieved_results = self.retriever.search(query, top_k=top_k)

        if pretty:
            return self.formatter.format_pretty(query, retrieved_results)
        return self.formatter.format_json(query, retrieved_results)

    def full_build(self):
        self.load_pdfs()
        self.make_chunks()
        self.embed()
        self.build_faiss()
        print("\n[✓] PIPELINE READY.\n")
