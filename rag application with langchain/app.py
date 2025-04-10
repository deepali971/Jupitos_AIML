from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize components
def initialize_rag():
    # Load and split documents (supports both PDF and text files)
    file_path = "data/Market_Segmentation_Analysis.pdf"  # Using the market segmentation analysis
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create vector store with Google embeddings
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    )
    
    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create RAG chain with memory using Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_version="v1beta")
    retriever = vectorstore.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    
    return qa_chain

if __name__ == "__main__":
    qa_chain = initialize_rag()
    print("RAG system ready. Type 'exit' to quit.")
    
    while True:
        query = input("Your question: ")
        if query.lower() == 'exit':
            break
        
        result = qa_chain.invoke({"question": query})
        print("Answer:", result["answer"])
