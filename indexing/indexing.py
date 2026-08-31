import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from config import settings


# loading env vars
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("API KEY not defined")

def load_pdf():
    try:
        # load pdf
        pdf_path = Path(__file__).parent.parent / "data" / "attention.pdf"
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        return docs
    except Exception as e:
            raise RuntimeError("Error in pdf data load !") from e
    
def split_and_chunk(docs):
    try:
        if docs is None or len(docs) == 0: 
            raise ValueError("docs cannot be empty!!, Empty docs passed")
        else :
            #split and chunk doc
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
            chunks = splitter.split_documents(docs)
            print(len(chunks))
            return chunks
    except Exception as e:
            raise RuntimeError("Error in splitting data") from e
        
    
def persist_in_chroma(chunks):
    try:
        if chunks is None or len(chunks) == 0:
            raise ValueError("chunks empty!")
        #Embedding
        embedding = OpenAIEmbeddings(
            model=settings.embedding_model
        )
        #Vectorization and Vector DB store
        vector_store = Chroma(
            collection_name="basic_rag_pipeline",
            embedding_function=embedding,
            persist_directory="./chroma_db"
        )
        vector_store.add_documents(chunks)
        return vector_store
    except Exception as e:
        raise RuntimeError("Error in persistence in chroma db!!") from e
        
    
if __name__ == "__main__":
    docs = load_pdf()
    chunks = split_and_chunk(docs)
    vector_store = persist_in_chroma(chunks)


