from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader




def create_or_load_vector_store(embeddings,dir):
    loader = DirectoryLoader(dir, glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    vector_store = Chroma.from_documents(texts, embeddings, collection_name='annualreport',persist_directory="Vectorstores/documents")

    print("Vector Store Created.......")
    return vector_store


