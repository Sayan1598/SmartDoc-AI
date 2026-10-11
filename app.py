
import os
import hashlib
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load API key from the .env file
load_dotenv()

# -----------------------------
# 1. Page configuration
# -----------------------------
st.set_page_config(
    page_title="SmartDoc-AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 SmartDoc-AI")
st.subheader("Upload a PDF and ask questions about its contents")

st.write(
    "Upload a document, process it, and ask questions. "
    "Answers are generated using relevant passages from your PDF."
)

# -----------------------------
# 2. Load the embedding model
# -----------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -----------------------------
# 3. Store data for this session
# -----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_pdf_hash" not in st.session_state:
    st.session_state.processed_pdf_hash = None

if "processed_pdf_name" not in st.session_state:
    st.session_state.processed_pdf_name = None


# -----------------------------
# 4. Upload a PDF
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Select a PDF from your computer."
)

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    current_pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    st.info(f"Selected document: {uploaded_file.name}")

    if st.button("Process PDF", type="primary"):
        temp_path = None

        try:
            with st.spinner(
                "Reading your PDF and building the search index..."
            ):
                # Save the uploaded bytes temporarily so PyPDFLoader
                # can read the PDF from a file path.
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:
                    temp_file.write(pdf_bytes)
                    temp_path = temp_file.name

                # Read the PDF pages
                loader = PyPDFLoader(temp_path)
                pages = loader.load()

                if not pages:
                    st.error("No pages could be read from this PDF.")
                    st.stop()

                # Split page text into smaller passages
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )
                chunks = splitter.split_documents(pages)

                if not chunks or not any(
                    chunk.page_content.strip() for chunk in chunks
                ):
                    st.error(
                        "No readable text was found. "
                        "This may be a scanned PDF that needs OCR."
                    )
                    st.stop()

                # Convert passages into embeddings and build FAISS
                embeddings = get_embeddings()
                new_vectorstore = FAISS.from_documents(
                    chunks,
                    embeddings
                )

                # Save the new index in this browser session
                st.session_state.vectorstore = new_vectorstore
                st.session_state.processed_pdf_hash = current_pdf_hash
                st.session_state.processed_pdf_name = uploaded_file.name

                st.success("PDF processed successfully!")

                st.write(f"Pages read: {len(pages)}")
                st.write(f"Text passages created: {len(chunks)}")

        except Exception as error:
            st.error(f"Could not process this PDF: {error}")

        finally:
            # Remove the temporary PDF file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

# -----------------------------
# 5. Ask questions about the PDF
# -----------------------------
st.divider()
st.header("💬 Ask a question")

if uploaded_file is None:
    st.info("Upload a PDF first.")

elif (
    st.session_state.vectorstore is None
    or st.session_state.processed_pdf_hash
    != current_pdf_hash
):
    st.warning(
        "Click 'Process PDF' before asking questions. "
        "If you selected a different PDF, process it again."
    )

else:
    st.success(
        f"Ready to answer questions about "
        f"{st.session_state.processed_pdf_name}"
    )

    question = st.text_input(
        "Your question",
        placeholder="Example: What is the main topic of this document?"
    )

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")

        else:
            api_key = os.getenv("GROQ_API_KEY")

            if not api_key:
                st.error(
                    "GROQ_API_KEY was not found. "
                    "Check your .env file."
                )

            else:
                try:
                    with st.spinner("Searching the PDF and generating an answer..."):
                        # Find the four most relevant passages
                        relevant_docs = (
                            st.session_state.vectorstore.similarity_search(
                                question,
                                k=4
                            )
                        )

                        context = "\n\n".join(
                            doc.page_content
                            for doc in relevant_docs
                        )

                        # Tell the LLM to use the retrieved passages
                        prompt = ChatPromptTemplate.from_template(
                            """
You are SmartDoc-AI, a helpful PDF question-answering assistant.

Answer the user's question using only the document passages below.
If the passages do not contain enough information, say:
"I could not find the answer in the provided document passages."
Do not invent facts.

Document passages:
{context}

User question:
{question}

Answer:
"""
                        )

                        # Connect to Groq
                        llm = ChatGroq(
                            model="openai/gpt-oss-120b",
                            temperature=0,
                            api_key=api_key
                        )

                        chain = prompt | llm
                        response = chain.invoke(
                            {
                                "context": context,
                                "question": question
                            }
                        )

                    st.subheader("Answer")
                    st.write(response.content)

                    # Show where the retrieved passages came from
                    with st.expander("View source passages"):
                        for index, doc in enumerate(relevant_docs, start=1):
                            page_number = doc.metadata.get("page")

                            if page_number is not None:
                                source_label = f"Page {page_number + 1}"
                            else:
                                source_label = f"Source {index}"

                            st.markdown(f"**{source_label}**")
                            st.write(doc.page_content)

                            if index < len(relevant_docs):
                                st.divider()

                except Exception as error:
                    st.error(f"Could not generate an answer: {error}")

st.divider()
st.caption(
    "SmartDoc-AI | Built with Streamlit, LangChain, "
    "FAISS, Hugging Face Embeddings, and Groq"
)
