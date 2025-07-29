# Semantic Spotter

## Libraries Used

- `os` - For environment variable and file path handling.
- `json` - For parsing JSON responses.
- `numpy` - For numerical operations and handling embeddings.
- `requests` - For making HTTP requests to APIs.
- `pdfplumber` - For extracting text from PDF files.
- `langchain` - For document schema, text splitting, embeddings, vector stores, chains, and LLM interfaces.
- `pathlib` - For handling file system paths.
- `dotenv` - For loading environment variables from `.env` files.

---

## Project Overview

Semantic Spotter is a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions based on insurance policy documents. It extracts text from PDFs, generates embeddings using a custom embedding API, stores them in a FAISS vector store, and uses a custom LLM API to generate answers based on retrieved relevant document chunks.

---

## Classes and Methods

### CustomAPIEmbeddings

#### 1. `embed_documents(self, texts)`
- **Input:** List of strings (`texts`), each representing a document or text chunk.
- **Process:** Calls the embedding API for each text and collects embeddings.
- **Output:** List of numpy arrays representing embeddings.

#### 2. `embed_query(self, text)`
- **Input:** Single string (`text`).
- **Process:** Calls the embedding API once for the input text.
- **Output:** Numpy array embedding.

#### 3. `get_embedding_from_api(self, text)`
- **Input:** Single string (`text`).
- **Process:**  
  - Prepares HTTP headers with API key.  
  - Sends POST request to embedding API with text and model info.  
  - Parses JSON response to extract embedding vector.  
- **Output:** Numpy array embedding.  
- **Raises:** `ValueError` if embedding is missing in response.

---

### CustomLLM

#### 1. `_call(self, prompt, stop=None)`
- **Input:**  
  - `prompt` (str): Text prompt for the LLM.  
  - `stop` (optional): Stop tokens (not used).  
- **Process:**  
  - Sends prompt to LLM API with conversation metadata.  
  - Parses streaming JSON response lines to reconstruct answer.  
- **Output:** Generated answer string.  
- **Raises:** `ValueError` if no answer is returned.

#### 2. `_identifying_params(self)`
- **Output:** Empty dictionary (used internally by LangChain).

#### 3. `_llm_type(self)`
- **Output:** String `"custom"` identifying the LLM type.

---

### InsuranceRAGChatbot

#### 1. `__init__(self, pdf_path)`
- **Input:**  
  - `pdf_path` (str): Path to insurance PDF document.  
- **Process:**  
  - Loads and splits PDF text into chunks.  
  - Loads or generates embeddings and builds FAISS vector store.  
  - Initializes custom LLM and RetrievalQA chain.  
- **Output:** Initialized chatbot instance.

#### 2. `load_and_split_pdf(self, pdf_path)`
- **Input:**  
  - `pdf_path` (str): Path to PDF file.  
- **Process:**  
  - Extracts text from all PDF pages.  
  - Splits text into chunks using RecursiveCharacterTextSplitter.  
  - Wraps chunks into Document objects.  
- **Output:** List of Document objects.  
- **Additional:** Prints number of chunks created.

#### 3. `chat(self, query)`
- **Input:**  
  - `query` (str): User question.  
- **Process:**  
  - Passes query to RetrievalQA chain for retrieval and generation.  
- **Output:** Dictionary with `"answer"` key containing the response.  
- **Fallback:** Default message if no answer found.

---

## Environment Variables

The project expects the following environment variables to be set (e.g., in `.env` or specified path):

- `EMBEDDING_URL` - URL endpoint for the embedding API.
- `CONVERSATION_URL` - URL endpoint for the LLM API.
- `API_TOKEN` - API key/token for authorization.

---

## Usage

1. Place your insurance PDF file in the project directory or specify the path.
2. Run the chatbot script:
   ```bash
   python insurance_langchain.py