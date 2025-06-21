from langchain_core.documents import Document
import chromadb
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd

def ingest(st):
    uploaded_files = st.file_uploader("📄 Upload your files", type=["pdf", "csv", "txt"], accept_multiple_files=True)
    # Handle uploaded files
    if uploaded_files:
        all_texts = []

        for uploaded_file in uploaded_files:
            ext = uploaded_file.name.split('.')[-1].lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            if ext == "pdf":
                reader = PyPDFLoader(tmp_path)
                pdf_docs = reader.load()
                content = "\n".join(doc.page_content for doc in pdf_docs if hasattr(doc, "page_content") and doc.page_content)
                all_texts.append(Document(page_content=content, metadata={"source": uploaded_file.name}))

            elif ext == "csv":
                df = pd.read_csv(tmp_path)
                content = df.to_csv(index=False)
                all_texts.append(Document(page_content=content, metadata={"source": uploaded_file.name}))

            elif ext == "txt":
                with open(tmp_path, "r", encoding="utf-8") as f:
                    content = f.read()
                all_texts.append(Document(page_content=content, metadata={"source": uploaded_file.name}))

        # Now split all collected documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(all_texts)

        # Create or load Chromadb collection
        client = chromadb.PersistentClient(path="E:/python/programs/basic_rag/output")
        # Remove existing collection if present
        if "my_collection" in [col.name for col in client.list_collections()]:
            client.delete_collection(name="my_collection")
        collection = client.create_collection(name="my_collection")
        collection.upsert(
            documents=[d.page_content for d in docs],
            ids=[str(i) for i in range(len(docs))]
        )
        return collection