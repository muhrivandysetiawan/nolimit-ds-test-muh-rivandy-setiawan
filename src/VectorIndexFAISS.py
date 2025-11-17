class VectorIndexFAISS:
    def __init__(self, metric='l2', use_gpu=False, gpu_device=0):
        assert metric in ('l2', 'cosine'), "metric must be 'l2' or 'cosine'"
        self.metric = metric
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device

        self.index = None
        self.dimension = None
        self.embeddings = None
        self.chunk_texts = None
        self.chunk_meta = None

    def _make_index(self, dim):
        if self.metric == 'l2':
            index = faiss.IndexFlatL2(dim)
        else:
            index = faiss.IndexFlatIP(dim)
        return index

    def _maybe_move_to_gpu(self, index):
        """Move index to GPU if requested and available."""
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, self.gpu_device, index)
                print(f"[FAISS] Index moved to GPU:{self.gpu_device}")
            except Exception as e:
                print(f"[FAISS] GPU move failed, falling back to CPU. Error: {e}")
        return index

    def build(self, embeddings, chunk_texts, chunk_meta, normalize=False):
        assert isinstance(embeddings, np.ndarray), "embeddings must be numpy array"
        assert embeddings.ndim == 2, "embeddings must be 2D array (n, dim)"
        n, dim = embeddings.shape
        assert len(chunk_texts) == n and len(chunk_meta) == n, "chunk_texts and chunk_meta must align with embeddings"

        self.dimension = dim
        self.embeddings = embeddings.astype('float32')
        self.chunk_texts = list(chunk_texts)
        self.chunk_meta = list(chunk_meta)

        if normalize or self.metric == 'cosine':
            faiss.normalize_L2(self.embeddings)

        index = self._make_index(self.dimension)
        index = self._maybe_move_to_gpu(index)

        index.add(self.embeddings)
        self.index = index

        print(f"[FAISS] Built index: vectors={self.index.ntotal}, dim={self.dimension}")
        return self.index

    def add(self, new_embeddings, new_texts, new_meta, normalize=False):
        assert self.index is not None, "Index not built. Call build() first."
        assert new_embeddings.shape[1] == self.dimension, "Dimension mismatch."

        emb = new_embeddings.astype('float32')
        if normalize or self.metric == 'cosine':
            faiss.normalize_L2(emb)

        self.index.add(emb)

        if self.embeddings is None:
            self.embeddings = emb.copy()
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

        self.chunk_texts.extend(new_texts)
        self.chunk_meta.extend(new_meta)

        print(f"[FAISS] Added {len(new_texts)} vectors. Total now: {self.index.ntotal}")
        return self.index.ntotal

    def search(self, query_embeddings, top_k=5, return_distance=True):
        assert self.index is not None, "Index not built."
        q_emb = query_embeddings.astype('float32')
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        assert q_emb.shape[1] == self.dimension, "Query embedding dimension mismatch."

        if self.metric == 'cosine':
            faiss.normalize_L2(q_emb)

        distances, indices = self.index.search(q_emb, top_k)

        results_all = []
        for qi in range(distances.shape[0]):
            row = []
            for pos, idx in enumerate(indices[qi]):
                if idx < 0 or idx >= len(self.chunk_texts):
                    continue
                score = float(distances[qi, pos])
                row.append({
                    "faiss_index": int(idx),
                    "score": score,
                    "chunk": self.chunk_texts[idx],
                    "metadata": self.chunk_meta[idx]
                })
            results_all.append(row)
        return results_all

    def save(self, dirpath):
        os.makedirs(dirpath, exist_ok=True)
        idx_path = os.path.join(dirpath, "index.faiss")
        try:
            cpu_index = faiss.index_cpu_to_all_gpus(self.index) if False else self.index
        except Exception:
            cpu_index = self.index
        try:
            cpu_index_for_write = faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index
        except Exception:
            cpu_index_for_write = self.index

        faiss.write_index(cpu_index_for_write, idx_path)
        if self.embeddings is not None:
            np.save(os.path.join(dirpath, "embeddings.npy"), self.embeddings)
        if self.chunk_meta is not None:
            with open(os.path.join(dirpath, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(self.chunk_meta, f, ensure_ascii=False, indent=2)
        if self.chunk_texts is not None:
            with open(os.path.join(dirpath, "chunks.txt"), "w", encoding="utf-8") as f:
                for c in self.chunk_texts:
                    f.write(c.replace("\n", " ") + "\n")

        print(f"[FAISS] Saved index+metadata to {dirpath}")


    def load(self, dirpath):

        idx_path = os.path.join(dirpath, "index.faiss")
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"No index.faiss found at {idx_path}")

        index = faiss.read_index(idx_path)
        self.index = index
        self.dimension = index.d

        emb_path = os.path.join(dirpath, "embeddings.npy")
        if os.path.exists(emb_path):
            self.embeddings = np.load(emb_path)

        meta_path = os.path.join(dirpath, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.chunk_meta = json.load(f)

        chunks_path = os.path.join(dirpath, "chunks.txt")
        if os.path.exists(chunks_path):
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunk_texts = [line.strip() for line in f]

        print(f"[FAISS] Loaded index from {dirpath}. Vectors={self.index.ntotal}, dim={self.dimension}")


    def move_to_gpu(self, gpu_device=0):
        if self.index is None:
            raise Exception("No index to move. Build or load index first.")
        try:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, gpu_device, self.index)
            self.use_gpu = True
            self.gpu_device = gpu_device
            print(f"[FAISS] Index moved to GPU:{gpu_device}")
        except Exception as e:
            raise RuntimeError(f"Failed to move FAISS index to GPU: {e}")
