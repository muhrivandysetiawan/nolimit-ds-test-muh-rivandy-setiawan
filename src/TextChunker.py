class TextChunker:

    DEFAULT_SECTION_HEADINGS = [
        "abstract", "introduction", "related work", "background",
        "method", "methods", "methodology", "approach", "model",
        "experiments", "experimental setup", "setup", "evaluation",
        "results", "analysis", "discussion", "conclusion", "conclusions",
        "future work", "limitations", "references", "acknowledgements",
        "acknowledgments", "appendix"
    ]

    def __init__(self, sentences_per_chunk=4, overlap_sentences=1, headings=None, verbose=True):
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        self.headings = headings or self.DEFAULT_SECTION_HEADINGS
        self.verbose = verbose

        self.chunks = []
        self.chunk_meta = []

    def _find_headings(self, text):
        matches = []
        pattern = r'(?i)\b(' + "|".join(re.escape(h) for h in self.headings) + r')\b'
        for m in re.finditer(pattern, text):
            label = m.group(1).strip()
            matches.append((m.start(), m.end(), label.lower()))
        matches.sort(key=lambda x: x[0])
        return matches

    def _section_boundaries(self, text):
        matches = self._find_headings(text)

        if not matches:
            return [("body", 0, len(text))]

        spans = []
        for i, (s, e, label) in enumerate(matches):
            start = s
            end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
            sec_name = label.lower()
            spans.append((sec_name, start, end))
        first_start = spans[0][1]
        if first_start > 0:
            spans.insert(0, ("preface", 0, first_start))
        return spans

    def _extract_section_text(self, text, start, end):
        return text[start:end].strip()

    def _split_sentences(self, text):
        sents = sent_tokenize(text)
        sents = [re.sub(r'\s+', ' ', s).strip() for s in sents if s.strip()]
        return sents

    def _chunk_sentences_in_section(self, sentences, doc_name, doc_id, section_name, base_chunk_id):
        chunks = []
        meta = []
        start = 0
        end = self.sentences_per_chunk
        chunk_id = base_chunk_id

        while start < len(sentences):
            chunk_sents = sentences[start:end]
            chunk_text = " ".join(chunk_sents).strip()
            if chunk_text:
                chunks.append(chunk_text)
                meta.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "source": doc_name,
                    "section": section_name,
                    "sentence_start": start,
                    "sentence_end": min(end, len(sentences)),
                })
                chunk_id += 1

            start = end - self.overlap_sentences
            end = start + self.sentences_per_chunk

        return chunks, meta, chunk_id

    def process_documents(self, doc_map):
        self.chunks = []
        self.chunk_meta = []

        for doc_id, (doc_name, full_text) in enumerate(doc_map.items()):
            if self.verbose:
                print(f"[Doc {doc_id}] Section-aware chunking: {doc_name}")

            text = full_text if full_text else ""
            spans = self._section_boundaries(text)

            if self.verbose:
                print(f" → Detected {len(spans)} sections")

            next_chunk_id = 0
            for sec_name, s_idx, e_idx in spans:
                sec_text = self._extract_section_text(text, s_idx, e_idx)
                sents = self._split_sentences(sec_text)
                if not sents:
                    continue

                sec_chunks, sec_meta, next_chunk_id = self._chunk_sentences_in_section(
                    sents, doc_name, doc_id, sec_name, next_chunk_id
                )

                self.chunks.extend(sec_chunks)
                self.chunk_meta.extend(sec_meta)

                if self.verbose:
                    print(f"   - Section '{sec_name}': {len(sec_chunks)} chunks")

            if self.verbose:
                print(f" → Total chunks for doc: {sum(1 for m in self.chunk_meta if m['doc_id']==doc_id)}\n")

        if self.verbose:
            print(f"All documents processed. Total chunks: {len(self.chunks)}")
        return self.chunks

    def get_chunks(self):
        return self.chunks

    def get_chunk_meta(self):
        return self.chunk_meta
