import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
loader = PyMuPDFLoader("./data/data_tuyensinh.pdf")
documents = loader.load()

if os.path.exists("index_tuyensinh"):
    vectorstore = FAISS.load_local("index_tuyensinh", embeddings=OpenAIEmbeddings(),allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(documents=documents, embedding=OpenAIEmbeddings())
    vectorstore.save_local("index_tuyensinh")
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})