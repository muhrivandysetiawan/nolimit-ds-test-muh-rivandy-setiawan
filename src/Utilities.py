import os
import re
from pypdf import PdfReader
from google.colab import files
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import time
import json
