class EmbeddingGenerator:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32, use_auth_token=None):

        print(f"Loading embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name, use_auth_token=use_auth_token)
        print("Model loaded successfully!\n")

        self.batch_size = batch_size

        self.embeddings = None
        self.chunk_texts = None
        self.chunk_meta = None

    def encode_chunks(self, chunk_texts, chunk_meta):
        print("Encoding chunks into embeddings...")

        all_embeddings = []
        num_chunks = len(chunk_texts)

        t0 = time.time()

        for start_idx in range(0, num_chunks, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_chunks)
            batch = chunk_texts[start_idx:end_idx]

            batch_emb = self.model.encode(batch, convert_to_numpy=True)
            all_embeddings.append(batch_emb)

            print(f"  Encoded batch {start_idx} → {end_idx}")

        self.embeddings = np.vstack(all_embeddings).astype("float32")
        self.chunk_texts = chunk_texts
        self.chunk_meta = chunk_meta

        print("\nEmbedding generation complete!")
        print(f"Shape: {self.embeddings.shape}")
        print(f"Time taken: {time.time() - t0:.2f} seconds\n")

        return self.embeddings

    def get_embeddings(self):
        if self.embeddings is None:
            raise Exception("No embeddings generated yet.")
        return self.embeddings

    def get_chunk_texts(self):
        return self.chunk_texts

    def get_chunk_meta(self):
        return self.chunk_meta
