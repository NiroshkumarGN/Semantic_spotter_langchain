# Import the necessary library
import os
import json
import numpy as np
import requests
import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


## Embedding and Conversational URL
env_path = Path(r"C:\Users\fg722f\Documents\Hackathon\graph-backend\AI_env.env")
load_dotenv(dotenv_path=env_path)
EMBEDDING = os.getenv("EMBEDDING_URL")
EMBEDDING_API_URL = f"{EMBEDDING}"
LLM_URL = os.getenv("CONVERSATION_URL")
LLM_API_URL = f"{LLM_URL}"
API_KEY = os.getenv("API_TOKEN")

EMBEDDINGS_FILE = "insurance_embeddings.npy"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


# Custom Embeddings class wrapping your embedding API
class CustomAPIEmbeddings(Embeddings):

    """
      Custom Embeddings class that interfaces with an external embedding API.

      Methods:
      - embed_documents(texts): Returns embeddings for a list of texts.
      - embed_query(text): Returns embedding for a single query text.
      - get_embedding_from_api(text): Helper method to call the embedding API and parse the response.
      """

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            emb = self.get_embedding_from_api(text)
            embeddings.append(emb)
        return embeddings

    def embed_query(self, text):
        return self.get_embedding_from_api(text)

    def get_embedding_from_api(self, text):
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'basic {API_KEY}'
        }
        data = {
            'input': text,
            'model': 'text-embedding-3-small'
        }
        response = requests.post(EMBEDDING_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        data = result.get('data')
        if data and len(data) > 0:
            embedding = data[0].get('embedding')
            if embedding:
                return np.array(embedding, dtype=np.float32)
        raise ValueError("Embedding not found in API response")

# Custom LLM class wrapping your LLM API
class CustomLLM(LLM):

    """
        Custom LLM class that interfaces with an external LLM API.

        Methods:
        - _call(prompt, stop=None): Sends a prompt to the LLM API and returns the generated response.
        - _identifying_params: Returns identifying parameters for the LLM (empty here).
        - _llm_type: Returns a string identifier for the LLM type.
        """

    def _call(self, prompt, stop=None):
        headers = {
            'accept': 'application/json',
            'Authorization': f'basic {API_KEY}',
            'Content-Type': 'application/json'
        }
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "model": "gpt-4o-mini",
            "conversation_mode": ["default"],
            "conversation_guid": "your_conversation_guid",
            "conversation_source": "bcai-api-system-identifier"
        }
        response = requests.post(LLM_API_URL, headers=headers, json=data)
        response.raise_for_status()
        raw_text = response.text

        # Parse streaming JSON lines if needed
        lines = raw_text.strip().split('\n')
        answer_parts = []
        for line in lines:
            if not line.strip():
                continue
            chunk = json.loads(line)
            choices = chunk.get('choices', [])
            if not choices:
                continue
            messages = choices[0].get('messages', [])
            if not messages:
                continue
            delta = messages[0].get('delta', '')
            if delta:
                answer_parts.append(delta)

        answer = ''.join(answer_parts).strip()
        if not answer:
            raise ValueError("No answer returned from LLM API.")
        return answer

    @property
    def _identifying_params(self):
        return {}

    @property
    def _llm_type(self) -> str:
        return "custom"

# Main chatbot class
class InsuranceRAGChatbot:
    """
        Main chatbot class implementing Retrieval-Augmented Generation (RAG) for insurance documents.

        Methods:
        - __init__(pdf_path): Loads and processes the PDF, creates or loads embeddings, and initializes the QA chain.
        - load_and_split_pdf(pdf_path): Extracts text from the PDF and splits it into manageable chunks.
        - chat(query): Accepts a user query, performs retrieval and generation, and returns the answer.
        """

    def __init__(self, pdf_path: str):
        self.documents = self.load_and_split_pdf(pdf_path)
        texts = [doc.page_content for doc in self.documents]

        self.embeddings = CustomAPIEmbeddings()

        # Load or create FAISS vector store
        if os.path.exists(EMBEDDINGS_FILE):
            print("Loading cached embeddings...")
            embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True)
            self.vectorstore = FAISS.from_embeddings(zip(texts, embeddings), self.embeddings)
        else:
            print("Generating embeddings and building vector store...")
            self.vectorstore = FAISS.from_texts(texts, self.embeddings)
            np.save(EMBEDDINGS_FILE, np.array(self.vectorstore.index.reconstruct_n(0, self.vectorstore.index.ntotal)))

        self.llm = CustomLLM()
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.vectorstore.as_retriever())

    def load_and_split_pdf(self, pdf_path):
        all_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        text_chunks = splitter.split_text(all_text)
        documents = [Document(page_content=chunk) for chunk in text_chunks]
        print(f"Loaded and split PDF into {len(documents)} chunks.")
        return documents

    def chat(self, query: str):
        result = self.qa_chain.invoke({"query": query})
        answer = result.get("result", "Sorry, I don't know the answer to that.")
        return {"answer": answer}


# Example usage
if __name__ == "__main__":
    pdf_path = "Principal-Sample-Life-Insurance-Policy.pdf"
    print(f"Loading PDF from: {pdf_path}")
    chatbot = InsuranceRAGChatbot(pdf_path)

    print("Insurance Langchain Chatbot is ready. Type your questions (type 'exit' to quit).")
    while True:
        user_query = input("You: ")
        if user_query.lower() in ['exit', 'quit']:
            print("Exiting chatbot.")
            break
        response = chatbot.chat(user_query)
        print("Bot:", response["answer"])