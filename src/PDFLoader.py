class PDFLoader:
    def __init__(self):
        self.file_names = []
        self.documents = []
        self.doc_map = {}

    def upload_pdfs(self):
        uploaded = files.upload()
        self.file_names = sorted(uploaded.keys())

        print("\nUploaded files (sorted):")
        for fn in self.file_names:
            print(" -", fn)
        return self.file_names

    def clean_text(self, text):
        text = text.replace("\u00a0", " ")

        text = re.sub(r'-\s*\n\s*', '', text)

        text = re.sub(r'\n+', ' ', text)

        text = re.sub(r'\[\d+\]', ' ', text)

        text = re.sub(r'[^\x00-\x7F]+', ' ', text)

        text = re.sub(r'(?<!\.)\.\.(?!\.)', '.', text)

        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def extract_text(self, pdf_path):
        reader = PdfReader(pdf_path)
        text = ""

        for i, page in enumerate(reader.pages):
            try:
                extracted = page.extract_text() or ""
                cleaned = self.clean_text(extracted)
                text += cleaned + " "
            except Exception as e:
                print(f"[Warning] Page {i} failed: {e}")

        return text.strip()

    # ----------------------------------------------------------
    def load_documents(self):
        if not self.file_names:
            raise Exception("No PDF uploaded. Run upload_pdfs() first.")

        print("\nExtracting & deep-cleaning documents...\n")

        self.documents = []
        self.doc_map = {}

        for fn in self.file_names:
            print(f"Processing: {fn}")
            fulltext = self.extract_text(fn)
            self.documents.append(fulltext)
            self.doc_map[fn] = fulltext
            print(f" → Cleaned length: {len(fulltext)} characters.\n")

        print("All documents loaded + cleaned successfully.")
        return self.documents

    def get_documents(self):
        return self.documents

    def get_doc_map(self):
        return self.doc_map
