class Retriever:
    def __init__(
        self,
        embedding_model,
        vector_index,
        top_k=5,
        rewrite_query=False
    ):
        self.model = embedding_model
        self.index = vector_index
        self.top_k = top_k
        self.rewrite_query = rewrite_query

        self.section_priority = {
            "abstract": 3.0,
            "introduction": 2.5,
            "method": 2.2,
            "methods": 2.2,
            "approach": 2.0,
            "model": 1.8,
            "analysis": 1.8,
            "results": 1.6,
            "experiments": 1.4,
            "evaluation": 1.3,
            "conclusion": 1.3,
            "discussion": 1.2,
        }

        self.skip_sections = {"references", "acknowledgments", "appendix"}

    def normalize_query(self, query):
        query = query.strip()
        query = re.sub(r"\s+", " ", query)
        return query.lower()

    def rewrite(self, query):
        return query
    def search(self, query, top_k=None):
        if top_k is None:
            top_k = self.top_k

        clean_q = self.normalize_query(query)
        if self.rewrite_query:
            clean_q = self.rewrite(clean_q)
        q_emb = self.model.encode(
            [clean_q],
            convert_to_numpy=True
        ).astype("float32")
        raw_results = self.index.search(q_emb, top_k * 3)
        if isinstance(raw_results, list) and len(raw_results) == 1 and isinstance(raw_results[0], list):
            raw_results = raw_results[0]
        reranked = self._rerank(raw_results)

        return reranked[:top_k]

    def _rerank(self, results):
        enhanced = []

        for item in results:

            if isinstance(item, dict):
                meta = item["metadata"]
                chunk = item["chunk"]
                faiss_distance = item["score"]
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                faiss_distance, idx = item
                meta = self.index.chunk_meta[idx]
                chunk = self.index.chunk_texts[idx]
            else:
                print("Unknown result format:", item)
                continue

            section = meta["section"].lower()
            if section in self.skip_sections:
                continue
            sim = 1 / (1 + faiss_distance)
            sec_bonus = self.section_priority.get(section, 1.0)
            final_score = sim * sec_bonus

            enhanced.append({
                "chunk": chunk,
                "metadata": meta,
                "distance": faiss_distance,
                "similarity": sim,
                "final_score": final_score
            })

        enhanced = sorted(enhanced, key=lambda x: x["final_score"], reverse=True)
        return enhanced
