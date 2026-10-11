
# SmartDoc-AI 📚

A Retrieval-Augmented Generation (RAG) application that lets users
upload PDF documents and ask questions about their contents.

## Features

- Upload PDF documents through a Streamlit interface
- Extract text from PDF pages
- Split text into smaller chunks
- Generate embeddings using Hugging Face Sentence Transformers
- Retrieve relevant passages using FAISS
- Generate context-based answers using a Groq-hosted language model
- Display source passages and page numbers

## Tech Stack

- Python
- Streamlit
- LangChain
- Hugging Face Sentence Transformers
- FAISS
- Groq API
- PyPDF

## How It Works

1. Upload a PDF.
2. Extract and split the document text.
3. Convert text chunks into embeddings.
4. Retrieve relevant passages for a question.
5. Generate an answer using the retrieved context.
6. Display the answer and source passages.

## Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd SmartDoc-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Never commit your real API key.

### 5. Run the application

```bash
python -m streamlit run app.py --server.fileWatcherType none
```

Open http://localhost:8501 in your browser.

## Limitations

- Answers depend on the passages retrieved from the PDF.
- Scanned PDFs may require OCR.
- The model can still produce incorrect answers.
- The current application keeps its uploaded document index in the active session.

## Future Improvements

- Chat history
- Improved retrieval and answer evaluation
- OCR support for scanned documents
- Persistent document storage
- Automated tests

## Author

Sayan Bachar
